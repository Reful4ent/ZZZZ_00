"""Manifest build, freeze and load (spec §4.1, §4.6–4.8, D-05, D-07).

Layout: <manifest_dir>/<version>/{datasets.json, images.parquet, completeness_report.json,
duplicates_report.json, exclusions.json, p1|p2|p3/pairs.parquet, manifest.frozen.json}.
manifest.frozen.json is written last; a version is never rewritten.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from expcore import registry
from expcore.config import Config, DatasetCfg
from expcore.errors import ConfigError, DataValidationError, ProtocolViolationError
from expcore.hashing import canonical_json, sha256_file, sha256_text, table_content_hash
from expcore.integrity import assert_not_blocking, check_integrity, write_duplicates_report
from expcore.io import dump_json, load_json, read_parquet, write_parquet
from expcore.scanners import check_counts, hpatches_completeness, scan
from expcore.schemas import (IMAGES_COLUMNS, IMAGES_NULLABLE, PAIRS_COLUMNS, PAIRS_NULLABLE, validate_datasets_json,
                             validate_images, validate_pairs)
from expcore.splits.hpatches import split_hpatches
from expcore.splits.kadid import split_kadid
from expcore.splits.mvtec import split_mvtec_category
from expcore.splits.sop import split_sop

FROZEN = "manifest.frozen.json"
PROGRAMS = ("p1", "p2", "p3")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_PROVENANCE_REQUIRED = ("source_url", "archive_name", "download_date", "license", "license_file")
_PROVENANCE_OPTIONAL = ("archive_path", "archive_sha256", "extra_archives")


@dataclass(frozen=True)
class ManifestInfo:
    version: str
    manifest_hash: str
    dir: Path


@dataclass(frozen=True)
class LoadedManifest:
    version: str
    dir: Path
    manifest_hash: str
    datasets: list[dict]
    images: pd.DataFrame
    pairs: Mapping[str, pd.DataFrame]


def assign_splits(raw: pd.DataFrame, dataset_cfg: DatasetCfg) -> pd.DataFrame:
    """Add `split` by the dataset's rule (§4.2–4.5); the unit of splitting is source_group_id."""
    raw = raw.copy()
    groups = raw["source_group_id"]
    by_group: dict[str, str] = {}
    if dataset_cfg.dataset_id == "kadid10k":
        parts = split_kadid(sorted(set(groups)), dataset_cfg.n_validation)
    elif dataset_cfg.dataset_id == "hpatches":
        parts = split_hpatches(sorted(set(groups)))
    elif dataset_cfg.dataset_id == "mvtec_ad":
        parts = {}
        train = raw[raw["role"] == "train_good"]
        for _, g in train.groupby("category_id", sort=True):
            for part, keys in split_mvtec_category(list(g["source_group_id"])).items():
                parts.setdefault(part, []).extend(keys)
        parts["real_test"] = list(raw.loc[raw["role"] == "test", "source_group_id"])   # official test, whole
    else:   # sop: official train classes split 80/20, official test unchanged
        train_classes = sorted({int(c) for c in raw.loc[raw["role"] == "train", "class_id"]})
        parts = {p: [str(c) for c in cs] for p, cs in split_sop(train_classes).items()}
        parts["test"] = list(raw.loc[raw["role"] == "test", "source_group_id"])
    for part, keys in parts.items():
        by_group.update(dict.fromkeys(keys, part))
    raw["split"] = groups.map(by_group)
    if raw["split"].isna().any():
        raise DataValidationError(f"{dataset_cfg.dataset_id}: images left without a split: "
                                  f"{raw.loc[raw['split'].isna(), 'relative_path'].head().tolist()}")
    return raw


def manifest_hash(images: pd.DataFrame, datasets: list[dict], pairs: Mapping[str, pd.DataFrame] | None) -> str:
    """D-07: content hash of the tables (not of parquet bytes) plus datasets.json."""
    parts = {
        "images": table_content_hash(images, list(IMAGES_COLUMNS), ["image_id"]),
        "datasets": sha256_text(canonical_json(sorted(datasets, key=lambda d: d["dataset_id"]))),
        "pairs": {p: table_content_hash(df, list(PAIRS_COLUMNS), ["pair_id"]) for p, df in sorted((pairs or {}).items())},
    }
    return sha256_text(canonical_json(parts))


def write_pairs(pairs: pd.DataFrame, images: pd.DataFrame, manifest_dir: Path, version: str, program: str) -> None:
    """Validate INV-02/03 and write <version>/<program>/pairs.parquet; refused once the version is frozen."""
    if program not in PROGRAMS:
        raise ValueError(f"program must be one of {PROGRAMS}, got {program!r}")
    vdir = Path(manifest_dir) / version
    path = vdir / program / "pairs.parquet"
    if (vdir / FROZEN).exists() or path.exists():
        raise ProtocolViolationError(f"{path}: manifest version {version!r} is frozen or pairs already written")
    validate_pairs(pairs, images)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(pairs[list(PAIRS_COLUMNS)], path, schema=PAIRS_COLUMNS, nullable=PAIRS_NULLABLE)


def _version(cfg: Config) -> str:
    versions = sorted({ds.manifest_version for ds in cfg.datasets.values()})
    if len(versions) != 1:
        raise ConfigError(f"datasets.*.manifest_version: one manifest holds all datasets, got {versions}")
    if not _VERSION.fullmatch(versions[0]):
        raise ConfigError(f"datasets.*.manifest_version: {versions[0]!r} is not a safe directory name")
    return versions[0]


def _apply_overrides(raws: dict[str, pd.DataFrame], overrides: Mapping | None) -> list[dict]:
    """exclude_image: [{sha256, reason}] removes every image with that hash before splitting (§4.6)."""
    if not overrides:
        return []
    if not isinstance(overrides, Mapping) or set(overrides) - {"exclude_image"}:
        raise ConfigError(f"overrides: only 'exclude_image' is supported, got {sorted(overrides)}")
    exclusions = []
    for i, entry in enumerate(overrides.get("exclude_image") or []):
        if not isinstance(entry, Mapping) or set(entry) != {"sha256", "reason"} \
                or not isinstance(entry["reason"], str) or not entry["reason"].strip():
            raise ConfigError(f"overrides.exclude_image[{i}]: expected {{sha256, reason}} with a non-empty reason")
        hits = {ds: raw["sha256"] == entry["sha256"] for ds, raw in raws.items()}
        paths = sorted(p for ds, hit in hits.items() for p in raws[ds].loc[hit, "relative_path"])
        if not paths:
            raise ConfigError(f"overrides.exclude_image[{i}]: no image with sha256 {entry['sha256']}")
        for ds, hit in hits.items():
            raws[ds] = raws[ds][~hit]
        exclusions.append({"sha256": entry["sha256"], "reason": entry["reason"], "relative_paths": paths})
    return exclusions


def _counts(images: pd.DataFrame, check: dict, excluded: int) -> dict[str, Any]:
    as_dict = lambda s: {str(k): int(v) for k, v in sorted(s.value_counts().items())}  # noqa: E731
    return {"images": len(images), "groups": int(images["source_group_id"].nunique()),
            "classes": int(images["class_id"].nunique()), "splits": as_dict(images["split"]),
            "categories": as_dict(images["category_id"].dropna()), "check_counts": check, "excluded": excluded}


def _dataset_record(ds: DatasetCfg, data_root: Path, counts: dict, completeness: dict | None) -> dict:
    prov = dict(ds.provenance)
    missing = [k for k in _PROVENANCE_REQUIRED if k not in prov]
    unknown = [k for k in prov if k not in _PROVENANCE_REQUIRED + _PROVENANCE_OPTIONAL]
    if missing or unknown:
        raise ConfigError(f"datasets.{ds.dataset_id}.provenance: missing {missing}, unknown {unknown}")
    archive_sha, verified = prov.get("archive_sha256"), False
    if prov.get("archive_path"):
        archive = data_root / prov["archive_path"]
        if not archive.is_file():
            raise DataValidationError(f"{archive}: archive named in provenance is missing")
        actual = sha256_file(archive)
        if archive_sha is not None and archive_sha != actual:
            raise DataValidationError(f"{archive}: SHA-256 {actual} differs from recorded {archive_sha}")
        archive_sha, verified = actual, True
    license_file = None
    if prov["license_file"] is not None:
        lic = data_root / prov["license_file"]
        if not lic.is_file():
            raise DataValidationError(f"{lic}: license file named in provenance is missing")
        license_file = {"path": prov["license_file"], "sha256": sha256_file(lic)}
    record = {"dataset_id": ds.dataset_id, "source_url": prov["source_url"], "archive_sha256": archive_sha,
              "license": prov["license"], "discovered_counts": counts, "manifest_version": ds.manifest_version,
              "download_date": prov["download_date"], "archive_name": prov["archive_name"],
              "archive_verified": verified, "license_file": license_file,
              "extra_archives": list(prov.get("extra_archives") or []),
              "completeness": completeness["status"] if completeness else None}
    if completeness and completeness["status"] == "subset":
        record["subset_name"] = completeness["subset_name"]
    return record


def build_manifests(cfg: Config, data_root: Path, manifest_dir: Path, overrides: Mapping | None = None) -> ManifestInfo:
    data_root = Path(data_root)
    version = _version(cfg)
    vdir = Path(manifest_dir) / version
    if vdir.exists():
        raise ProtocolViolationError(f"{vdir}: manifest version {version!r} already exists; versions are never "
                                     "rewritten — choose a new manifest_version")
    raws, checks, completeness = {}, {}, {}
    for ds_id, ds in sorted(cfg.datasets.items()):
        raw = scan(ds, data_root)
        checks[ds_id] = check_counts(raw, ds)   # composition of the delivery, before exclusions
        if ds_id == "hpatches":
            rep = hpatches_completeness(raw, ds, data_root)
            if rep["status"] == "subset" and not rep["subset_name"]:
                raise ConfigError("datasets.hpatches.expected.subset_name: required, HPatches delivery is a subset "
                                  f"(incomplete: {rep['incomplete']})")
            completeness[ds_id] = rep
        raw["sha256"] = [sha256_file(data_root / p) for p in raw["relative_path"]]
        raws[ds_id] = raw
    n_before = {ds_id: len(raw) for ds_id, raw in raws.items()}
    exclusions = _apply_overrides(raws, overrides)

    frames = []
    for ds_id, raw in raws.items():
        raw = assign_splits(raw, cfg.datasets[ds_id])
        ordinal = {p: i for i, p in enumerate(sorted(raw["relative_path"]))}   # D-05: code-point order
        raw["image_id"] = [f"{ds_id}:{ordinal[p]:06d}" for p in raw["relative_path"]]
        frames.append(raw)
    images = pd.concat(frames, ignore_index=True)[list(IMAGES_COLUMNS)]
    images = images.sort_values("image_id", kind="mergesort").reset_index(drop=True)
    validate_images(images)
    report = check_integrity(images)
    datasets = [_dataset_record(cfg.datasets[ds_id], data_root,
                                _counts(images[images["dataset_id"] == ds_id], checks[ds_id],
                                        n_before[ds_id] - len(raws[ds_id])),
                                completeness.get(ds_id))
                for ds_id in sorted(raws)]
    validate_datasets_json(datasets)

    vdir.mkdir(parents=True)
    dump_json(datasets, vdir / "datasets.json")
    write_parquet(images, vdir / "images.parquet", schema=IMAGES_COLUMNS, nullable=IMAGES_NULLABLE)
    dump_json(completeness, vdir / "completeness_report.json")
    dump_json(exclusions, vdir / "exclusions.json")
    write_duplicates_report(report, vdir / "duplicates_report.json")
    assert_not_blocking(report)   # blocking overlaps: reports stay for inspection, the version is not frozen
    pairs = {}
    for program, builder in registry.pair_builders().items():
        pairs[program] = builder(images, cfg)[list(PAIRS_COLUMNS)]
        write_pairs(pairs[program], images, manifest_dir, version, program)
    h = manifest_hash(images, datasets, pairs)
    dump_json({"version": version, "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "manifest_hash": h}, vdir / FROZEN)
    return ManifestInfo(version, h, vdir)


def load_manifest(manifest_dir: Path, version: str, *, verify: bool = True) -> LoadedManifest:
    vdir = Path(manifest_dir) / version
    if not (vdir / FROZEN).is_file():
        raise DataValidationError(f"{vdir}: manifest version {version!r} is not frozen (no {FROZEN}); "
                                  "run build-manifests")
    frozen = load_json(vdir / FROZEN)
    datasets = load_json(vdir / "datasets.json")
    images = read_parquet(vdir / "images.parquet")
    pairs = {p: read_parquet(vdir / p / "pairs.parquet") for p in PROGRAMS if (vdir / p / "pairs.parquet").is_file()}
    if verify:
        validate_datasets_json(datasets)
        validate_images(images)
        for table in pairs.values():
            validate_pairs(table, images)
        actual = manifest_hash(images, datasets, pairs)
        if frozen.get("version") != version or actual != frozen.get("manifest_hash"):
            raise DataValidationError(f"{vdir}: content hash {actual} differs from {FROZEN} "
                                      f"({frozen.get('manifest_hash')}) — manifest modified or corrupted")
    return LoadedManifest(version, vdir, frozen["manifest_hash"], datasets, images, pairs)
