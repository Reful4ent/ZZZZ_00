import importlib.util

import pytest

from expcore.primitives import PRIMITIVES, PrimitiveUnavailable, primitive_versions, register_primitive, run_check
from expcore.runlog import collect_environment

pytestmark = pytest.mark.unit

HAS_TF = importlib.util.find_spec("tensorflow") is not None


@pytest.mark.parametrize("cid", ["PRIM-SIFT", "PRIM-COLOR", "PRIM-LINALG"])
def test_primitive_check_passes(cid):
    run_check(cid)


@pytest.mark.tf
@pytest.mark.skipif(not HAS_TF, reason="TensorFlow is not installed (extra `metric`)")
def test_adam_first_step():        # θ₁ = θ₀ − lr·ĝ/(|ĝ|+ε)
    run_check("PRIM-ADAM")


@pytest.mark.tf
@pytest.mark.skipif(not HAS_TF, reason="TensorFlow is not installed (extra `metric`)")
def test_backbone_shape():
    run_check("PRIM-BACKBONE")


def test_registry_lists_all_checks_and_versions_in_env():
    env_prims = collect_environment(deterministic_ops=None)["library_primitives"]
    assert {(p["name"], p["check_id"]) for p in PRIMITIVES} <= {(p["name"], p["check_id"]) for p in env_prims}
    assert {p["check_id"] for p in PRIMITIVES} == {"PRIM-SIFT", "PRIM-COLOR", "PRIM-LINALG", "PRIM-ADAM",
                                                   "PRIM-BACKBONE"}
    for p in primitive_versions():
        assert set(p) == {"name", "library", "version", "check_id"}
    assert next(p for p in primitive_versions() if p["library"] == "numpy")["version"]


@pytest.mark.skipif(HAS_TF, reason="checks the behaviour without TensorFlow")
def test_tf_checks_skip_with_reason_when_missing():
    for cid in ("PRIM-ADAM", "PRIM-BACKBONE"):
        with pytest.raises(PrimitiveUnavailable, match="TensorFlow"):
            run_check(cid)


def test_unknown_check_and_registration():
    with pytest.raises(KeyError):
        run_check("PRIM-NOPE")
    calls = []
    register_primitive("toy", "numpy", "PRIM-TOY-1", lambda: calls.append(1))
    run_check("PRIM-TOY-1")
    assert calls == [1] and any(p["check_id"] == "PRIM-TOY-1" for p in primitive_versions())
