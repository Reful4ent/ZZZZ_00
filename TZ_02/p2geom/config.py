"""Конфигурация конвейера P2 geometric (ТЗ 02).

Значения по умолчанию — начальный проектный профиль ТЗ (раздел 4, 5, 7).
Они нужны для smoke-прогонов и не считаются оптимальными: рабочий профиль
выбирается калибровкой на validation и затем замораживается (см. calibration.py).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class FeatureConfig:
    method: str = "sift"            # sift | orb | superpoint
    max_keypoints: int = 2000       # ГЕОМ-02: базовый лимит
    max_side: int = 1024            # длинная сторона рабочего изображения (только уменьшение)
    # SuperPoint (ГЕОМ-03): метаданные весов заполняет пользователь.
    superpoint_weights: str = ""
    superpoint_weights_url: str = (
        "https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth"
    )
    superpoint_weights_sha256: str = ""
    superpoint_weights_license: str = "MagicLeap SuperPointPretrainedNetwork license (non-commercial research)"
    superpoint_weights_commit: str = ""
    superpoint_conf_thresh: float = 0.015
    superpoint_nms_dist: int = 4
    superpoint_border: int = 4


@dataclass(frozen=True)
class MatchConfig:
    ratio: float = 0.75             # τr; validation-кандидаты {0.65, 0.75, 0.85}


@dataclass(frozen=True)
class RansacConfig:
    estimator: str = "own"          # own (обязательная ветвь Г2-106) | opencv (baseline)
    eps_g: float = 3.0              # εg, пиксели исходного E; sensitivity {1, 3, 5}
    max_trials: int = 5000
    confidence: float = 0.999
    adaptive: bool = True           # адаптивное сокращение бюджета, всегда протоколируется
    seed: int = 2026
    refine: bool = True
    refine_max_nfev: int = 100
    w_eps: float = 1e-12            # |w| <= w_eps: нормализация однородных координат запрещена
    collinear_tol: float = 1e-3     # синус угла, ниже которого тройка точек считается коллинеарной


@dataclass(frozen=True)
class GeometryGateConfig:
    n_min: int = 12
    rho_min: float = 0.25
    eps_med: float = 2.0
    c_I_min: float = 0.10
    c_E_min: float = 0.10
    cond_max: float = 1e10
    w_min: float = 1e-8
    use_support: bool = True        # n_in, ρ_in, медиана ошибки (абляция P2-E4)
    use_coverage: bool = True       # c_I, c_E (абляция P2-E4)
    # Допускаемый диапазон отношения площадей перенесённого контура; задаётся генератором
    # только в синтетических опытах (None — не проверять).
    area_ratio_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class AlignConfig:
    rho_vis_min: float = 0.50
    a_min: int = 1024
    border_px: int = 0              # исключаемые границы E при построении Ω_obj


@dataclass(frozen=True)
class CompareConfig:
    grid: int = 8                   # sensitivity: 4, 8, 16
    v_min: float = 0.80
    n_min_area: int = 16
    tau_pix_grid: tuple[float, ...] = (2.0, 5.0, 10.0)
    region_stat: str = "mean_delta"   # mean_delta (Г2-118) | delta_of_means (P2-E5)
    compensation: bool = False      # Г2-116, дополнительный режим
    cal_band: float = 0.10
    a_bounds: tuple[float, float] = (0.8, 1.2)
    b_bounds: tuple[float, float] = (-0.1, 0.1)
    critical: str = "none"          # none | center4 (K_req = K_crit)


@dataclass(frozen=True)
class DecisionConfig:
    tau_pix: float = 5.0
    tau_d: float = 5.0              # τd (τ_area в Г2-121)
    tau_a: float = 0.10             # τa (τ_frac в Г2-121)
    tau_B: float = 0.10             # t_loc
    tau_D: float = 5.0              # t_glob
    use_B: bool = True              # абляция P2-E6
    use_critical: bool = True


@dataclass(frozen=True)
class Config:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    matching: MatchConfig = field(default_factory=MatchConfig)
    ransac: RansacConfig = field(default_factory=RansacConfig)
    gate: GeometryGateConfig = field(default_factory=GeometryGateConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    compare: CompareConfig = field(default_factory=CompareConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=list).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def replace(self, **sections) -> "Config":
        """replace(gate={'n_min': 8}) — частичная замена полей секций."""
        new = {}
        for name, patch in sections.items():
            new[name] = dataclasses.replace(getattr(self, name), **patch)
        return dataclasses.replace(self, **new)


_SECTIONS = {
    "features": FeatureConfig, "matching": MatchConfig, "ransac": RansacConfig,
    "gate": GeometryGateConfig, "align": AlignConfig, "compare": CompareConfig,
    "decision": DecisionConfig,
}


def config_from_dict(d: dict) -> Config:
    kwargs = {}
    for name, cls in _SECTIONS.items():
        sec = dict(d.get(name, {}) or {})
        unknown = set(sec) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown keys in section {name}: {sorted(unknown)}")
        for f in dataclasses.fields(cls):
            if f.name in sec and isinstance(sec[f.name], list):
                sec[f.name] = tuple(sec[f.name])
        kwargs[name] = cls(**sec)
    unknown = set(d) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    return Config(**kwargs)


def load_config(path: str | Path | None) -> Config:
    if path is None:
        return Config()
    with open(path, encoding="utf-8") as fh:
        return config_from_dict(yaml.safe_load(fh) or {})


def save_config(cfg: Config, path: str | Path) -> None:
    def plain(x):
        if isinstance(x, dict):
            return {k: plain(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [plain(v) for v in x]
        return x
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(plain(cfg.to_dict()), fh, allow_unicode=True, sort_keys=False)
