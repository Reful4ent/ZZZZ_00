"""Fake predictors for runner tests (T-REP-03)."""

from expcore.decisions import Decision, ScoreDirection
from expcore.registry import Outcome

HIGHER = ScoreDirection.HIGHER_IS_MATCH


class RejectAllPredictor:
    method_id = "reject_all"

    def predict(self, pair, images, ctx):
        return Decision(Outcome.REJECT, None, None, HIGHER, "REJECT_GENERIC", {})


class RandomScorePredictor:
    """Scores from the unit's stream only, so results do not depend on resume history."""
    method_id = "random_score"

    def predict(self, pair, images, ctx):
        images.load(pair.query_id, "gray")
        s = float(ctx.rng.random())
        ctx.augmentations.log(ctx.unit_id, pair.pair_id, "noop", {"s": round(s, 6)})
        return Decision(Outcome.DECIDED, int(s >= 0.5), s, HIGHER, None, {"inliers": s * 10})


class OracleLabelPredictor:
    """Uses ground truth explicitly given to it (oracle mode is explicit, §5.3)."""
    method_id = "oracle_label"

    def __init__(self, labels):
        self._labels = dict(labels)

    def predict(self, pair, images, ctx):
        y = self._labels[pair.pair_id]
        return Decision(Outcome.DECIDED, y, float(y), HIGHER, None, {})


class CrashOnUnitPredictor(RandomScorePredictor):
    """RandomScorePredictor that raises inside one unit of work."""

    def __init__(self, crash_unit):
        self._crash_unit = crash_unit

    def predict(self, pair, images, ctx):
        if ctx.unit_id == self._crash_unit:
            raise RuntimeError("simulated algorithm failure")
        return super().predict(pair, images, ctx)


class BatchQueryPredictor:
    method_id = "batch_query"

    def predict(self, queries, images, ctx):
        return [Decision(Outcome.DECIDED, 1, float(i), HIGHER, None, {}) for i, _ in enumerate(queries)]
