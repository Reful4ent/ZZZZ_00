"""Strict `params` schema of module metric (spec 11) and the profile validator. TF-free: imported by metric.cli."""
import dataclasses
from dataclasses import dataclass
from typing import Any

from expcore.config import Config, register_params_schema
from expcore.errors import ConfigError

METHOD_IDS = ("proxy_cal", "base_minimal", "contrastive", "triplet", "classifier", "imagenet_fixed", "rgb_hist",
              "proxy_softmax_noL2")
# keys the experiments vary (spec 11); lr / margin only for methods whose grid (8.4) contains them.
# normalize_l2 is not varied: E6 no-L2 is the method proxy_softmax_noL2 (M-D22)
VARIED_KEYS = frozenset({"lambda", "K", "n_warm", "lr", "margin", "train_augment",
                         "per_class_limit", "cohort_min_images", "steps_per_epoch", "degradation"})
GRID_KEYS = {"contrastive": {"lr", "margin"}, "triplet": {"lr", "margin"}, "classifier": {"lr"}}
SMOKE_KEYS = frozenset({"weights", "resize_short", "crop"})   # M-D27 substitutions
WEIGHTS_SHA256 = "66c8b43daff3fcc15bc4f30e3d2a167e21a14d9c9598a5394e5516471f4af504"   # metric.model.weights_file_sha256()

BASE_PARAMS: dict[str, Any] = {
    "model": {"backbone": "resnet50", "weights": "imagenet", "weights_sha256": WEIGHTS_SHA256,
              "embedding_dim": 128, "bn_mode": "frozen_stats_trainable_affine"},
    "input": {"resize_short": 256, "crop": 224, "train_augment": True, "flip_p": 0.5, "normalize_l2": True},
    "batch": {"classes": 16, "per_class": 2},
    "optimizer": {"name": "adam", "lr": 1.0e-4, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1.0e-7},
    "train": {"epochs": 40, "steps_per_epoch": "auto", "precision": "fp32"},
    "proxy": {"kappa": 16, "init": "normal_then_normalize", "eps_p": 1.0e-12},
    "embed": {"eps_z": 1.0e-12},
    "calibration": {"lambda": 0.0, "queues": True, "K": 32, "n_warm": 4},
    "queue": {"check_every": 1000, "sum_tol": 1.0e-6},
    "loss": {"margin": None},
    "data": {"class_subset": None, "per_class_limit": None, "cohort_min_images": None},
    "checkpoint": {"every_steps": None},
    "search": {"block_q": 1024, "block_g": 8192, "k": [1, 5, 10], "d2_tol": 1.0e-6},
    "pairs": {"max_pos_per_class": 20, "max_attempts": 100},
    "threshold": {"criterion_id": "max_recall_at_fpr_le_0.05", "fpr_max": 0.05},
    "degradation": "none",
}


@dataclass(frozen=True)
class ModelP:
    backbone: str
    weights: str | None
    weights_sha256: str
    embedding_dim: int
    bn_mode: str


@dataclass(frozen=True)
class InputP:
    resize_short: int
    crop: int
    train_augment: bool
    flip_p: float
    normalize_l2: bool


@dataclass(frozen=True)
class BatchP:
    classes: int
    per_class: int


@dataclass(frozen=True)
class OptimizerP:
    name: str
    lr: float
    beta_1: float
    beta_2: float
    epsilon: float


@dataclass(frozen=True)
class TrainP:
    epochs: int
    steps_per_epoch: Any   # "auto" or int (checked by validate_profile)
    precision: str


@dataclass(frozen=True)
class ProxyP:
    kappa: float
    init: str
    eps_p: float


@dataclass(frozen=True)
class EmbedP:
    eps_z: float


@dataclass(frozen=True, init=False, repr=False, eq=False)
class CalibrationP:
    # "lambda" is a Python keyword, so the field exists only in __annotations__ and the init takes **kwargs;
    # expcore's loader/asdict then see the plain key "lambda" without changes to expcore
    # (generated __repr__/__eq__ would need `self.lambda`, so they are written by hand).
    __annotations__ = {"lambda": float, "queues": bool, "K": int, "n_warm": int}

    def __init__(self, **kw):
        if set(kw) != set(self.__annotations__):
            raise TypeError(f"CalibrationP fields {sorted(self.__annotations__)}, got {sorted(kw)}")
        for k, v in kw.items():
            object.__setattr__(self, k, v)

    def __eq__(self, other):
        return type(other) is type(self) and vars(self) == vars(other)

    def __repr__(self):
        return f"CalibrationP({vars(self)})"


@dataclass(frozen=True)
class QueueP:
    check_every: int
    sum_tol: float


@dataclass(frozen=True)
class LossP:
    margin: float | None


@dataclass(frozen=True)
class ClassSubsetP:
    n_classes: int
    seed: int   # own seed (spec 4.5): the subset must not depend on the run seed


@dataclass(frozen=True)
class DataP:
    class_subset: ClassSubsetP | None
    per_class_limit: int | None
    cohort_min_images: int | None


@dataclass(frozen=True)
class CheckpointP:
    every_steps: int | None


@dataclass(frozen=True)
class SearchP:
    block_q: int
    block_g: int
    k: tuple[int, ...]
    d2_tol: float


@dataclass(frozen=True)
class PairsP:
    max_pos_per_class: int
    max_attempts: int


@dataclass(frozen=True)
class ThresholdP:
    criterion_id: str
    fpr_max: float


@dataclass(frozen=True)
class MetricParams:
    model: ModelP
    input: InputP
    batch: BatchP
    optimizer: OptimizerP
    train: TrainP
    proxy: ProxyP
    embed: EmbedP
    calibration: CalibrationP
    queue: QueueP
    loss: LossP
    data: DataP
    checkpoint: CheckpointP
    search: SearchP
    pairs: PairsP
    threshold: ThresholdP
    degradation: str


register_params_schema("metric", MetricParams)


def params_dict(p: MetricParams) -> dict:
    """The spec-11 keyed dict that train/eval consume."""
    return dataclasses.asdict(p)


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        out.update(_flat(v, f"{prefix}{k}.") if isinstance(v, dict) and k != "class_subset" else {f"{prefix}{k}": list(v) if isinstance(v, tuple) else v})
    return out


def validate_profile(cfg: Config) -> None:
    if len(cfg.algorithms) != 1:
        raise ConfigError(f"algorithms: exactly one method_id required, got {sorted(cfg.algorithms)}")
    (method,) = cfg.algorithms
    if method not in METHOD_IDS:
        raise ConfigError(f"algorithms: unknown method_id {method!r}; known: {METHOD_IDS}")
    p = params_dict(cfg.params)
    cal, spe = p["calibration"], p["train"]["steps_per_epoch"]
    for key, ok, rule in (("calibration.lambda", lambda v: v >= 0, ">= 0"), ("calibration.K", lambda v: v >= 1, ">= 1"),
                          ("calibration.n_warm", lambda v: v >= 1, ">= 1"), ("optimizer.lr", lambda v: v > 0, "> 0"),
                          ("optimizer.epsilon", lambda v: v > 0, "> 0"), ("proxy.kappa", lambda v: v > 0, "> 0"),
                          ("proxy.eps_p", lambda v: v > 0, "> 0"), ("embed.eps_z", lambda v: v > 0, "> 0"),
                          ("model.embedding_dim", lambda v: v >= 1, ">= 1"),
                          ("loss.margin", lambda v: v is None or v > 0, "> 0 or null")):
        sec, name = key.split(".")
        if not ok(p[sec][name]):
            raise ConfigError(f"{key}: must be {rule}, got {p[sec][name]!r}")
    if cal["n_warm"] > cal["K"]:
        raise ConfigError(f"calibration.n_warm={cal['n_warm']} must not exceed K={cal['K']}")
    if not cal["queues"] and method == "proxy_cal":
        raise ConfigError("calibration.queues=false is allowed only for base_minimal and baselines")
    if not (spe == "auto" or (isinstance(spe, int) and not isinstance(spe, bool) and spe > 0)):
        raise ConfigError(f"train.steps_per_epoch: must be 'auto' or a positive integer, got {spe!r}")
    if cfg.profile == "confirmatory" and p["data"]["class_subset"] is not None:
        raise ConfigError("data.class_subset is forbidden in the confirmatory profile")

    allowed = set(VARIED_KEYS) | {"class_subset", "queues"}   # queues: checked above
    allowed -= {"lr", "margin"} - GRID_KEYS.get(method, set())
    if cfg.profile == "smoke":
        allowed |= SMOKE_KEYS | {"lr", "margin"}
    base = _flat(BASE_PARAMS)
    bad = [k for k, v in _flat(p).items() if v != base[k] and k.rsplit(".", 1)[-1] not in allowed]
    if bad:
        raise ConfigError(f"params: {bad} differ from the base profile (spec 5.1/11) and are not varied by experiments")
