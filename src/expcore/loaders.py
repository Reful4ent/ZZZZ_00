"""Input and evaluation views (spec §5.3, ДАН-04, INV-11) and the split guard (§7.4).

Algorithms get PairInput/QueryInput (whitelisted columns only) and ImageStore.load(image_id, color);
they never see paths, file names or evaluation columns.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd

from expcore.config import FORBIDDEN_PAIR_COLUMNS, OPTIONAL_PAIR_COLUMNS, REQUIRED_PAIR_COLUMNS
from expcore.errors import ConfigError, DataValidationError, ProtocolViolationError, TestAccessError
from expcore.schemas import EVAL_ONLY_PAIR_COLUMNS

PROFILE_SPLITS = {"pilot": frozenset({"train", "dev", "validation"})}
TEST_SPLITS = frozenset({"test", "synthetic_test", "real_test"})
# Whitelisted by D-17 but a path; paths can carry labels (D-05), so algorithms must not receive it as-is.
PATH_PAIR_COLUMNS = frozenset({"input_mask_path"})
_COLOR_FLAGS = {
    # IGNORE_ORIENTATION: manifest width/height come from IMREAD_UNCHANGED, which ignores EXIF orientation too.
    "bgr": cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
    "rgb": cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
    "gray": cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION,
}
_PAIR_KEYS = ("pair_id", "query_id", "reference_id", "relation_id")
_EVAL_KEYS = ("pair_id", "query_id", "reference_id", "relation_id", "split", "source_group_id")


@dataclass(frozen=True)
class PairInput:
    pair_id: str
    query_id: str
    reference_id: str
    relation_id: str
    extras: Mapping[str, Any]


@dataclass(frozen=True)
class QueryInput:
    query_id: str
    extras: Mapping[str, Any]


class SplitGuard:
    """pilot: train/dev/validation; confirmatory: test-like splits only after allow_test(); smoke+fixture: all."""

    def __init__(self, profile: str, *, fixture: bool) -> None:
        if profile not in ("smoke", "pilot", "confirmatory"):
            raise ConfigError(f"profile: unknown {profile!r}")
        self._profile = profile
        self._open = profile == "smoke" and fixture
        self._test_opened = False

    def allow_test(self) -> None:
        """Called only after protocol.open_test has recorded the opening (spec §7.5)."""
        if self._profile != "confirmatory":
            raise ProtocolViolationError(f"test splits can be opened only in confirmatory, not {self._profile}")
        self._test_opened = True

    def check(self, split: str) -> None:
        if self._open:
            return
        if split in TEST_SPLITS:
            if self._profile == "confirmatory" and self._test_opened:
                return
            raise TestAccessError(f"split {split!r} is not accessible in profile {self._profile}"
                                  + (" before the test is opened" if self._profile == "confirmatory" else ""))
        if self._profile != "confirmatory" and split not in PROFILE_SPLITS["pilot"]:
            raise TestAccessError(f"split {split!r} is not accessible in profile {self._profile}")


class ImageStore:
    def __init__(self, data_root: Path, images: pd.DataFrame, guard: SplitGuard) -> None:
        self._root = Path(data_root)
        self._guard = guard
        self._records = dict(zip(images["image_id"], zip(images["relative_path"], images["split"])))

    def load(self, image_id: str, color: Literal["bgr", "rgb", "gray"]) -> np.ndarray:
        if color not in _COLOR_FLAGS:
            raise ConfigError(f"color must be one of {sorted(_COLOR_FLAGS)}, got {color!r}")
        record = self._records.get(image_id)
        if record is None:
            raise DataValidationError(f"unknown image_id {image_id!r}")
        rel, split = record
        self._guard.check(split)
        try:
            data = np.fromfile(self._root / rel, dtype=np.uint8)   # works for non-ASCII paths on Windows
        except OSError:
            data = np.empty(0, np.uint8)
        img = cv2.imdecode(data, _COLOR_FLAGS[color]) if data.size else None
        if img is None:   # the message names the opaque id only: paths may carry labels (D-05)
            raise DataValidationError(f"image {image_id!r} cannot be read")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if color == "rgb" else img


def _check_columns(columns: Sequence[str]) -> list[str]:
    columns = list(columns)
    leaked = [c for c in columns if c in FORBIDDEN_PAIR_COLUMNS or c in EVAL_ONLY_PAIR_COLUMNS]
    if leaked:
        raise ConfigError(f"inputs.pair_columns: evaluation column(s) {leaked} are not visible to algorithms (INV-11)")
    if PATH_PAIR_COLUMNS & set(columns):
        raise ConfigError(f"inputs.pair_columns: {sorted(PATH_PAIR_COLUMNS & set(columns))} would hand a file path "
                          "to the algorithm (§5.3); opaque mask access is not implemented yet")
    unknown = [c for c in columns if c not in REQUIRED_PAIR_COLUMNS + OPTIONAL_PAIR_COLUMNS]
    missing = [c for c in REQUIRED_PAIR_COLUMNS if c not in columns]
    if unknown or missing:
        raise ConfigError(f"inputs.pair_columns: unknown {unknown}, missing {missing}")
    return columns


def _plain(v: Any) -> Any:
    if isinstance(v, np.generic):
        v = v.item()
    return None if isinstance(v, float) and math.isnan(v) else v


def load_pair_inputs(pairs: pd.DataFrame, columns: Sequence[str], split: str, guard: SplitGuard) -> list[PairInput]:
    """Input view of one split, sorted by pair_id; reads only whitelisted columns."""
    columns = _check_columns(columns)
    guard.check(split)
    extras = [c for c in columns if c not in _PAIR_KEYS]
    rows = pairs.loc[pairs["split"] == split, list(_PAIR_KEYS) + extras].sort_values("pair_id", kind="mergesort")
    return [PairInput(*(str(v) for v in row[:4]), {c: _plain(v) for c, v in zip(extras, row[4:])})
            for row in rows.itertuples(index=False, name=None)]


def load_pair_evaluation(pairs: pd.DataFrame, split: str) -> pd.DataFrame:
    """Evaluation view (y_match, gt_*) for thresholds, metrics, aggregate and oracle mode only."""
    cols = list(_EVAL_KEYS) + [c for c in EVAL_ONLY_PAIR_COLUMNS if c in pairs.columns]
    return pairs.loc[pairs["split"] == split, cols].sort_values("pair_id", kind="mergesort").reset_index(drop=True)
