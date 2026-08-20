"""No registry entry may carry a token cap that truncates the production prompt.

This exists because the caps were wrong by default and nobody noticed for
months. `vllm-local`, `ollama-local` and `local-gemma-4-31b` all shipped
``max_tokens: 1000`` — and `local-gemma-4-31b` shipped the literal "1000/60s
pair" that `local-gemma-4-26b`'s own comment, nine entries above it, records as
catastrophic: *"truncated mid-thought and then timed out."* The rejection was
written down next to the code that would have prevented it.

The measurement behind the floor, taken 2026-08-12 on 60 monolithic calls with
the production reasoning-first variant (`disconfirm_relnature_rf`) against
gemma-4-26b:

    output tokens: p50 574, p90 1507, p99 4353, max 4353
    would truncate at 1000: 10/60 = 16.7%  Wilson [0.093, 0.280]
    would truncate at 2048:  5/60 =  8.3%  Wilson [0.036, 0.181]
    would truncate at 4096:  1/60 =  1.7%  Wilson [0.003, 0.089]
    would truncate at 8192:  0/60 =  0.0%  Wilson [0.000, 0.060]

A truncated read is not a cheap failure: it costs the full wall clock, yields no
usable verdict, and on paths without the withholding policy it can contribute a
verdict recovered from mid-chain-of-thought text.

A cap is a CEILING, not a reservation — an entry that never emits 8k tokens pays
nothing for permitting them. So the floor applies to every entry rather than
only the ones measured.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indra_belief.model_client import LOCAL_MODELS  # noqa: E402

# p99 of the measured distribution is 4353. 8192 clears it with 1.9x headroom
# and truncated 0/60 in the sample that set it.
CAP_FLOOR = 8192

_CAPPED = sorted(
    (name, cfg["max_tokens"])
    for name, cfg in LOCAL_MODELS.items()
    if cfg.get("max_tokens") is not None
)


def test_registry_declares_caps_for_the_entries_that_need_them():
    """Guard the guard: if nothing declares max_tokens, the floor test is vacuous."""
    assert len(_CAPPED) >= 20, f"only {len(_CAPPED)} entries declare max_tokens"


def test_no_registry_entry_disappears_silently():
    """A vanished entry makes the parametrized floor test QUIETER, not redder.

    Earned 2026-08-13: a string edit to model_client.py matched the wrong
    `"max_tokens": 8192,` occurrence, deleted the whole `local-gemma-4-31b`
    header, and fused two entries. Python parsed the result, the registry
    imported fine, and the full suite reported 1780 passed — one case fewer than
    before, which is invisible in a pass count. Pinning the population makes a
    deletion fail loudly; adding a model is then a deliberate one-line update
    here, which is the right amount of friction for changing what we can serve.
    """
    assert len(LOCAL_MODELS) == 31, (
        f"registry has {len(LOCAL_MODELS)} entries, expected 31. If you ADDED a "
        "model, update this number. If you did not, an entry was deleted — check "
        "for an edit that matched the wrong anchor."
    )
    for required in ("local-gemma-4-26b", "local-gemma-4-31b", "vllm-gemma-4-26b"):
        assert required in LOCAL_MODELS, f"{required} vanished from the registry"

    # A RENAME must stay reachable under the old name. `vllm-local` was renamed
    # to `vllm-gemma-4-26b` because the belief profile registry is keyed on the
    # registry name with no served-model id, so a name that identifies only the
    # server lets a fitted profile follow that server onto different weights.
    # The correction must not break a collaborator's existing command line, and
    # every prior rename in this registry was alias-preserving.
    from indra_belief.model_client import canonical_model_name

    for old_name, canonical in (("vllm-local", "vllm-gemma-4-26b"),
                                ("ollama-local", "ollama-gemma-3-27b")):
        assert canonical_model_name(old_name) == canonical, (
            f"{old_name} no longer resolves; a rename dropped its alias"
        )


@pytest.mark.parametrize("name,cap", _CAPPED, ids=[n for n, _ in _CAPPED])
def test_no_registry_entry_truncates_the_production_prompt(name: str, cap: int):
    assert cap >= CAP_FLOOR, (
        f"{name} caps generation at {cap}, below the {CAP_FLOOR} floor. "
        f"Measured p90 is 1507 and max 4353 for the production reasoning-first "
        f"prompt, so this silently truncates long deliberations. Raise it — a "
        f"ceiling costs nothing when it is not reached."
    )


def test_the_three_fitted_readers_keep_the_caps_their_profiles_were_measured_under():
    """Raising a cap changes the reader; a fitted profile is only valid for the
    reader it was measured on. These three must not drift silently."""
    from indra_belief.calibration_constants import _FITTED_CONFIGS

    expected = {
        "bedrock-gemma-4-26b": 32000,
        "remote-gemma-4-26b": 32000,
        "remote-medpsy-4b": 32000,
        # Fitted 2026-08-13 on the self-hosted MLX stack. Its whole fit and
        # validation ran at 8192, and 4 capped reads were withheld on that basis;
        # changing this cap changes the reader the profile describes.
        "local-gemma-4-26b": 8192,
    }
    profiled = {model for model, _sha in _FITTED_CONFIGS}
    assert profiled == set(expected), (
        f"the set of fitted readers changed ({profiled}); re-derive the caps "
        "their profiles were measured under before updating this test"
    )
    for model, cap in expected.items():
        assert LOCAL_MODELS[model]["max_tokens"] == cap, (
            f"{model} has a fitted calibration profile measured at max_tokens "
            f"{cap}; changing it to {LOCAL_MODELS[model]['max_tokens']} changes "
            "the reader and may invalidate the fit"
        )


def test_vllm_shard_runner_inherits_the_cap_instead_of_hardcoding_one():
    """The shard runner used to hardcode --max-tokens 1000, silently overriding
    the registry on every corpus-scale run. The ceiling belongs with the model."""
    import argparse
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_shards",
        Path(__file__).resolve().parent.parent / "scripts" / "run_vllm_processed_shards.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    parser: argparse.ArgumentParser = mod.build_parser()
    defaults = {a.dest: a.default for a in parser._actions}
    assert defaults["max_tokens"] is None, (
        "--max-tokens must default to None so the registry supplies the ceiling; "
        f"got {defaults['max_tokens']!r}"
    )
    assert defaults["timeout"] is None, (
        "--timeout must default to None so the registry supplies it; "
        f"got {defaults['timeout']!r}"
    )
