"""validate-data (spec §4.8): checks a frozen manifest against the data on disk; never modifies the manifest."""

from pathlib import Path

from expcore.config import Config
from expcore.errors import DataValidationError, ProtocolError
from expcore.hashing import sha256_file
from expcore.integrity import check_integrity
from expcore.io import dump_json, load_json
from expcore.manifests import FROZEN, assign_splits, load_manifest, manifest_hash, manifest_version
from expcore.runlog import now_utc
from expcore.scanners import scan
from expcore.schemas import validate_datasets_json, validate_images, validate_pairs

MAX_ITEMS = 200   # per issue in the report; the count is always exact


def report_path(cfg: Config) -> Path:
    return cfg.paths.out_root / cfg.profile / "validate-data" / manifest_version(cfg) / "validation_report.json"


def validate_data(cfg: Config, data_root: Path, manifest_dir: Path) -> dict:
    """Hash, files and their SHA-256, schemas/INV-01…04, recomputed splits, cross-split overlaps."""
    data_root = Path(data_root)
    version = manifest_version(cfg)
    m = load_manifest(manifest_dir, version, verify=False)   # not frozen → DataValidationError (exit 3)
    issues: list[dict] = []

    def issue(check: str, message: str, items=()) -> None:
        items = sorted(set(items))
        issues.append({"check": check, "message": message, "n": len(items), "items": items[:MAX_ITEMS]})

    frozen = load_json(m.dir / FROZEN)
    if manifest_hash(m.images, m.datasets, m.pairs) != frozen["manifest_hash"]:
        issue("manifest_hash", f"table contents differ from {FROZEN}")
    checks = [("datasets_schema", lambda: validate_datasets_json(m.datasets)),
              ("images_invariants", lambda: validate_images(m.images))]
    checks += [(f"pairs_{p}", lambda t=t: validate_pairs(t, m.images)) for p, t in sorted(m.pairs.items())]
    for name, check in checks:
        try:
            check()
        except DataValidationError as e:
            issue(name, str(e))

    missing, changed, actual = [], [], {}
    for rel, recorded in zip(m.images["relative_path"], m.images["sha256"]):
        path = data_root / rel
        if not path.is_file():
            missing.append(rel)
            continue
        actual[rel] = sha256_file(path)
        if actual[rel] != recorded:
            changed.append(rel)
    if missing:
        issue("files_present", "image files listed in the manifest are missing", missing)
    if changed:
        issue("files_sha256", "image files differ from their recorded SHA-256", changed)

    excluded = {p for e in load_json(m.dir / "exclusions.json") for p in e["relative_paths"]}
    drift = []
    for ds_id, ds in sorted(cfg.datasets.items()):
        saved = m.images[m.images["dataset_id"] == ds_id]
        saved = dict(zip(saved["relative_path"], saved["split"]))
        try:
            raw = scan(ds, data_root)
            raw = assign_splits(raw[~raw["relative_path"].isin(excluded)], ds)
        except ProtocolError as e:
            issue("splits_recomputed", f"{ds_id}: cannot recompute the split: {e}")
            continue
        recomputed = dict(zip(raw["relative_path"], raw["split"]))
        drift += [p for p in saved.keys() | recomputed.keys() if saved.get(p) != recomputed.get(p)]
    if drift:
        issue("splits_recomputed", "recomputed split differs from the saved manifest", drift)

    images = m.images.copy()
    images["sha256"] = [actual.get(p, s) for p, s in zip(images["relative_path"], images["sha256"])]
    overlaps = check_integrity(images)
    if overlaps["blocking"]:
        issue("cross_split_duplicates", "identical images or source groups in several parts",
              [e["sha256"] for e in overlaps["cross_split_sha256"]]
              + [e["source_group_id"] for e in overlaps["cross_split_groups"]])

    report = {"version": version, "manifest_hash": frozen["manifest_hash"], "checked_at": now_utc(),
              "n_images": len(m.images), "ok": not issues, "issues": issues}
    path = report_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(report, path)
    return report
