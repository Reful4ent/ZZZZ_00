"""Repo-wide test isolation for program packages (no TZ00 source involved)."""
import metric  # noqa: F401  (imported once at collection, so the pop below sticks for the session)
from expcore import registry


def pytest_collection_finish(session):
    # `import metric` registers the p3 pair builder process-wide (TZ03 spec 3.4 item 3); TZ00 tests build
    # manifests with no program builders, so every session starts without it. Metric tests install it explicitly
    # (tests.metric.world, real_pairs=True); registration itself is checked in a fresh interpreter
    # (tests/metric/unit/test_cli.py::test_pair_builder_registered).
    registry._PAIR_BUILDERS.pop("p3", None)
