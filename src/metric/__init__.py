"""P3 metric. Importing the package registers the p3 pair builder and its reason code (no TensorFlow)."""
from expcore.registry import Outcome, pair_builders, register_pair_builder, register_reason_code
from metric.pairs import p3_pair_builder
from metric.search import NO_POSITIVE

if "p3" not in pair_builders():   # a test may have installed its own p3 builder before the first import
    register_pair_builder("p3", p3_pair_builder)
register_reason_code(NO_POSITIVE, Outcome.NOT_APPLICABLE)
