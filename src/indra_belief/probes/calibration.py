"""Sentence-grain calibration for the direct verdict probe.

``ProbeReading.delta_logit`` is a log-odds measurement, not a probability.
This module is the apply boundary for the fitted mapping from that measurement
to ``p_hat = P(the reading is correct)``.  The persisted model is the existing
:class:`indra_belief.probe_combiner.FrozenCombiner` with one feature; no second
isotonic implementation lives here.

The calibration was fitted at the sentence/evidence grain.  It must not be
used as a statement-belief update.  Consumers that need additive evidence can
use ``weight_of_evidence`` — how far this one read moves the belief, in
log-odds relative to the fit-set base rate::

    weight_of_evidence = logit(p_hat) - logit(base_rate)

Endpoint probabilities are clipped by the combiner's shared ``to_logit``
policy, because an isotonic model can legitimately emit zero or one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

from indra_belief.probe_combiner import (
    LOGIT_EPS,
    FrozenCombiner,
    to_logit,
)
from indra_belief.probes.battery import probe_digest
from indra_belief.probes.reader import (
    DIRECT_PROBE_ID,
    IN_CALL_PROBE_ID,
    ProbeReading,
    read_probe,
)


CALIBRATION_FILENAME = "sentence_probe_calibration.json"
CALIBRATION_MODEL = "local-gemma-4-26b"
CALIBRATION_MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-8bit"
CALIBRATION_PROBE_DIGEST = (
    "2aa7729f9b4f5e897c6e99baf25956c710c1a36f4f49dfd7f89b4fc747d641ed"
)
SENTENCE_SCORE_CONTRACT_VERSION = 1
SENTENCE_SCORE_KIND = "calibrated_probability_correct"
DEFAULT_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "probe_battery"
    / CALIBRATION_FILENAME
)
# A calibration artifact carries exactly ONE feature, but it may be either
# route's reading of it. Kept as a set of ACCEPTED single-id tuples rather than
# widened to "any id": the check exists to refuse an artifact fitted on some
# other feature entirely, and dropping it would let a two-probe combiner load
# into a one-probe call site.
CALIBRATED_PROBE_IDS = (DIRECT_PROBE_ID,)
ACCEPTED_PROBE_IDS = ((DIRECT_PROBE_ID,), (IN_CALL_PROBE_ID,))


def _validate_probe_profile() -> None:
    current = probe_digest(DIRECT_PROBE_ID)
    if current != CALIBRATION_PROBE_DIGEST:
        raise ValueError(
            "direct sentence probe content does not match the fitted calibration "
            f"profile: expected {CALIBRATION_PROBE_DIGEST}, got {current}"
        )


# Measured on the fit corpus: the LOSING label lands at rank 42/83/168 of the
# top-k window. A client that declares less headroom than this can physically
# issue the probe but will lose a label on a large fraction of rows, so it is
# not a production reading client. `read_probe` keeps its own mechanical floor
# of 2 and raises ProbeTopKError per row, which is what a FITTING run wants.
MIN_PROBE_TOP_LOGPROBS = 256

# Serving identity -> fitted artifact. The key includes the SERVED model id, not
# just the registry name, because delta_logit magnitudes are substrate-specific:
# the same weights read in-process and over HTTP correlate at r=0.955 but differ
# 2.4x in range and disagree in sign on 10% of rows. An isotonic map fitted on
# one serving stack is therefore not valid on another, exactly as a reader
# confusion profile is not valid across prompts.
#
# To add a substrate: read raw delta_logits on it (which
# `probe_reading_supported` now permits without a calibration), fit an isotonic,
# ship the artifact, and add one row here. Adding a row is the whole change.
_SENTENCE_CALIBRATIONS: dict[tuple[str, str], str] = {
    (CALIBRATION_MODEL, CALIBRATION_MODEL_ID): CALIBRATION_FILENAME,
    (
            "vllm-gemma-4-26b",
            "google/gemma-4-26B-A4B-it",
    ): "incall_vllm.json",
}


def probe_reading_supported(client) -> bool:
    """Whether ``client`` can produce a ``delta_logit`` at all.

    A CAPABILITY question, deliberately separate from whether a calibration
    exists for this client. Fusing the two made the remedy unreachable: fitting
    a calibration for a new serving stack requires reading raw delta_logits on
    that stack, which an identity-pinned gate forbids.
    """

    config = getattr(client, "config", None)
    if not isinstance(config, Mapping):
        return False
    top_k = config.get("max_top_logprobs")
    return (
        getattr(client, "_guard", None) is None
        and getattr(client, "backend", "openai_compat") == "openai_compat"
        and isinstance(top_k, int)
        and not isinstance(top_k, bool)
        and top_k >= MIN_PROBE_TOP_LOGPROBS
    )


def sentence_calibration_path_for(client) -> Path | None:
    """The fitted artifact for this client's exact serving identity, or None."""

    config = getattr(client, "config", None)
    if not isinstance(config, Mapping):
        return None
    key = (getattr(client, "model_name", None), config.get("model_id"))
    filename = _SENTENCE_CALIBRATIONS.get(key)  # type: ignore[arg-type]
    if filename is None:
        return None
    return DEFAULT_CALIBRATION_PATH.parent / filename


def supports_sentence_calibration(client) -> bool:
    """Whether ``client`` can be read AND has a calibration fitted for it.

    The production gate: both halves must hold before a calibrated probability
    is emitted. An uncalibrated but capable client reads `False` here and still
    reads `True` from :func:`probe_reading_supported`.
    """

    if not probe_reading_supported(client):
        return False
    if sentence_calibration_path_for(client) is None:
        return False
    try:
        _validate_probe_profile()
    except ValueError:
        return False
    return True


def replace_sentence_score(
    result: Mapping[str, object],
    record: Mapping[str, object],
    client,
    *,
    record_id: str | None,
    enabled: bool | None = None,
    extra_probe_call: bool = False,
) -> dict[str, object]:
    """Attach whatever the probe can measure on THIS client, and nothing more.

    Two independent questions, answered by two existing predicates rather than by
    knowing anything about which model is on the other end:

      probe_reading_supported(client)        can it produce a delta_logit?
      sentence_calibration_path_for(client)  is an isotonic fitted for it?

    So the layers come off separately:

      ``probe_delta_logit``   RAW log-odds. Written whenever the client can be
                              read AT ALL, calibrated or not. It is a
                              measurement, not a score: comparable only within
                              one serving stack, and never a probability.
      ``score``               calibrated p_hat — only with a fitted artifact.
      ``weight_of_evidence``  the additive form belief consumes — same condition.

    WHY THE RAW LAYER EXISTS. Fitting a calibration for a new stack needs
    delta_logits FROM that stack, and gating the read on having a calibration
    made that circular: the first run on a new server collected nothing, so
    calibrating it required a second full pass. Now any run doubles as its own
    fitting corpus and calibration becomes an offline step over data already on
    disk.

    Model- and client-agnostic by construction. No model name appears here; the
    only stack-specific facts are DATA — ``max_top_logprobs`` in the registry and
    a row in ``_SENTENCE_CALIBRATIONS`` — so a new serving stack is two entries,
    not a code change.

    TWO WAYS TO GET THE MARGIN, and the cheap one is preferred.

    A variant whose output contract emits the verdict FIRST already has the
    margin in the response we just made — ``result["in_call_label_margin"]``.
    That is free and MEASURED BETTER (n=80: AUROC 0.8734 against the separate
    probe's 0.7237; within-verdict 0.7814 against 0.6856). It is always used
    when present.

    Otherwise a separate forced-position request can supply it, and that costs
    ONE EXTRA CALL PER EVIDENCE. It is therefore ``extra_probe_call``, opt-in.
    It used to fire unconditionally on any readable client, silently doubling
    the request count of every run against a probe-capable server — and it is
    redundant on a verdict-first variant, which reads the same quantity for
    nothing.
    """

    enriched = dict(result)
    enriched["score"] = None
    enriched["score_error"] = None
    enriched["weight_of_evidence"] = None
    enriched["probe_delta_logit"] = None

    if enabled is False:
        return enriched
    artifact = sentence_calibration_path_for(client)

    # Free path first: the scoring call already carried it.
    in_call = result.get("in_call_label_margin")
    if isinstance(in_call, (int, float)) and not isinstance(in_call, bool):
        enriched["probe_delta_logit"] = float(in_call)
        # RAW ONLY, on purpose. The shipped isotonic was fitted on PROBE deltas
        # and its knots span -1.70..+1.61; in-call deltas are ~3x wider
        # (MEASURED n=80: median |13.22| against the probe's |4.34|), so every
        # in-call reading lands off the end of the curve and saturates to 0 or 1.
        # A calibrated number produced that way is the verdict wearing false
        # precision — the same hazard as applying one serving stack's map to
        # another, across ACQUISITION ROUTES instead. The in-call route needs its
        # own isotonic; persisting the raw margin is what makes fitting one
        # possible, and until then no score is the truthful output.
        enriched["score_error"] = (
            "NoInCallCalibration: raw in-call margin persisted; the shipped "
            "isotonic was fitted on separate-probe deltas and does not transfer"
        )
        return enriched

    if not extra_probe_call:
        return enriched
    readable = probe_reading_supported(client) if enabled is None else True
    if not readable:
        return enriched
    if not record_id:
        enriched["score_error"] = (
            "ValueError: calibrated sentence score requires a source-hash identity"
        )
        return enriched
    if not str(record.get("evidence_text") or ""):
        return enriched

    try:
        reading = read_probe(record, client)
        enriched["probe_delta_logit"] = reading.delta_logit
        if artifact is not None:
            _validate_probe_profile()
            calibrated = calibrate_probe(
                reading, record_id=record_id, calibration=_calibration_at(artifact)
            )
            enriched["score"] = calibrated.p_hat
            enriched["weight_of_evidence"] = calibrated.weight_of_evidence
    except Exception as exc:
        enriched["score_error"] = f"{type(exc).__name__}: {exc}"
    return enriched


class CalibratedProbeReading(NamedTuple):
    """A calibrated correctness probability and its weight of evidence."""

    p_hat: float
    weight_of_evidence: float


def load_calibration(
    path: str | Path = DEFAULT_CALIBRATION_PATH,
) -> FrozenCombiner:
    """Reload and validate the persisted sentence-probe calibration."""

    _validate_probe_profile()
    artifact_path = Path(path)
    with artifact_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    calibration = FrozenCombiner.from_dict(payload)
    if calibration.probe_ids not in ACCEPTED_PROBE_IDS:
        raise ValueError(
            "sentence probe calibration must contain exactly one of "
            f"{ACCEPTED_PROBE_IDS!r}, got {calibration.probe_ids!r}"
        )
    return calibration


@lru_cache(maxsize=8)
def _calibration_at(path: Path) -> FrozenCombiner:
    """Load and cache one fitted artifact per serving identity.

    Keyed by path rather than a single global slot, because more than one
    substrate can be registered at once and each has its own isotonic map.
    """

    return load_calibration(path)


@lru_cache(maxsize=1)
def _default_calibration() -> FrozenCombiner:
    """Load the shipped frozen model once per serving process."""

    return load_calibration()


def _calibration_or_default(
    calibration: FrozenCombiner | None,
) -> FrozenCombiner:
    _validate_probe_profile()
    resolved = _default_calibration() if calibration is None else calibration
    if not isinstance(resolved, FrozenCombiner):
        raise TypeError("calibration must be a FrozenCombiner")
    if resolved.probe_ids not in ACCEPTED_PROBE_IDS:
        raise ValueError(
            "sentence probe calibration must contain exactly one of "
            f"{ACCEPTED_PROBE_IDS!r}, got {resolved.probe_ids!r}"
        )
    return resolved


def calibrated_probabilities(
    delta_logits,
    *,
    record_ids,
    calibration: FrozenCombiner | None = None,
) -> np.ndarray:
    """Map a batch of direct-probe log-odds to calibrated probabilities.

    ``record_ids`` is mandatory so the underlying ``FrozenCombiner`` can
    refuse rows used to fit its isotonic map.  The returned values, unlike the
    input ``delta_logits``, are probabilities and are safe to use for ECE or
    Brier scoring.
    """

    try:
        values = np.asarray(delta_logits, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "delta_logits must be coercible to a float vector"
        ) from exc
    if values.ndim != 1:
        raise ValueError("delta_logits must have shape (n,)")

    model = _calibration_or_default(calibration)
    return model.score(
        values.reshape(-1, 1),
        record_ids=record_ids,
        # The ARTIFACT'S own ids, not the direct probe's. Passing a constant
        # here meant an in-call artifact loaded fine and then died at score
        # time -- "X column order does not match probe_ids" -- for every row.
        # The loader was widened to accept either route's id without this being
        # widened with it, so the feature was reachable and non-functional, and
        # `apply_weights`' bare except turned that into a silent counter.
        probe_ids=model.probe_ids,
    )


def calibrated_probability(
    delta_logit: float,
    *,
    record_id: str,
    calibration: FrozenCombiner | None = None,
) -> float:
    """Map one direct-probe log-odds measurement to ``p_hat``."""

    scores = calibrated_probabilities(
        [delta_logit],
        record_ids=(record_id,),
        calibration=calibration,
    )
    return float(scores[0])


def weight_of_evidence(
    p_hat,
    base_rate,
    *,
    eps: float = LOGIT_EPS,
):
    """Return ``logit(p_hat) - logit(base_rate)`` with finite endpoints.

    Scalars produce a ``float`` and array-like inputs produce an ``ndarray``.
    Validation and clipping are delegated to the combiner's canonical
    :func:`to_logit`, keeping the evidence convention identical everywhere.
    """

    evidence = np.asarray(
        to_logit(p_hat, eps=eps) - to_logit(base_rate, eps=eps),
        dtype=float,
    )
    if evidence.ndim == 0:
        return float(evidence)
    return evidence


def calibrate_probe(
    reading: ProbeReading,
    *,
    record_id: str,
    calibration: FrozenCombiner | None = None,
) -> CalibratedProbeReading:
    """Calibrate a ``ProbeReading`` and expose its additive evidence form."""

    if not isinstance(reading, ProbeReading):
        raise TypeError("reading must be a ProbeReading")
    model = _calibration_or_default(calibration)
    p_hat = calibrated_probability(
        reading.delta_logit,
        record_id=record_id,
        calibration=model,
    )
    weight = weight_of_evidence(p_hat, model.fit_prevalence)
    return CalibratedProbeReading(p_hat=p_hat, weight_of_evidence=weight)


# ``calibrate_reading`` reads naturally beside ``read_probe`` while the more
# explicit spelling above keeps the domain visible at call sites.
calibrate_reading = calibrate_probe


__all__ = [
    "CALIBRATED_PROBE_IDS",
    "CALIBRATION_FILENAME",
    "CALIBRATION_MODEL",
    "CALIBRATION_MODEL_ID",
    "CALIBRATION_PROBE_DIGEST",
    "SENTENCE_SCORE_CONTRACT_VERSION",
    "SENTENCE_SCORE_KIND",
    "DEFAULT_CALIBRATION_PATH",
    "CalibratedProbeReading",
    "calibrate_probe",
    "calibrate_reading",
    "calibrated_probabilities",
    "calibrated_probability",
    "load_calibration",
    "replace_sentence_score",
    "supports_sentence_calibration",
    "weight_of_evidence",
]
