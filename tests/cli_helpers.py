"""A small end-to-end world for CLI tests: fixture data, a config, a frozen manifest with toy pairs."""

import copy
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from expcore import registry
from expcore.config import load_config
from expcore.manifests import build_manifests
from expcore.protocol import register_version
from tests.conftest import TEST_MODULE, smoke_cfg_dict, write_cfg
from tests.fixtures_builder import make_all, make_sop

RELATION = {"p1": "P1_SAME_SCENE", "p2": "P2_NO_CHANGE_IN_REGION", "p3": "P3_SAME_PRODUCT"}


def toy_pairs(program: str):
    """Pair consecutive images of each (dataset, split); y_match = 1 when they share a source group."""
    def build(images: pd.DataFrame, cfg) -> pd.DataFrame:
        rows = []
        for (dataset_id, split), g in images.sort_values("image_id").groupby(["dataset_id", "split"]):
            ids, groups = list(g.image_id), list(g.source_group_id)
            for i in range(len(ids) - 1):
                rows.append(dict(pair_id=f"{program}:{dataset_id}:{split}:{i:04d}", query_id=ids[i],
                                 reference_id=ids[i + 1], relation_id=RELATION[program],
                                 y_match=int(groups[i] == groups[i + 1]), split=split, source_group_id=groups[i],
                                 transform_id=None, severity=None, seed=None, gt_H_path=None, gt_mask_path=None,
                                 input_mask_path=None))
        return pd.DataFrame(rows)
    return build


@dataclass
class World:
    root: Path
    cfg_dict: dict
    cfg_path: Path
    registry_path: Path
    out_root: Path
    exp_dict: dict = None
    exp_path: Path = None
    runs: list = None

    def argv(self, *cmd: str) -> list[str]:
        """Module command line (module config)."""
        return [*cmd, "--config", str(self.cfg_path), "--protocol-registry", str(self.registry_path)]

    def exp_argv(self, *cmd: str) -> list[str]:
        """experiments command line (experiments config: same data sections, params.runs)."""
        return [*cmd, "--config", str(self.exp_path), "--protocol-registry", str(self.registry_path)]

    def write(self, name: str = "cfg.yaml") -> Path:
        self.cfg_path = write_cfg(self.cfg_dict, self.root / name)
        self.exp_dict = {**copy.deepcopy(self.cfg_dict), "params": {"runs": self.runs}}
        self.exp_path = write_cfg(self.exp_dict, self.root / f"exp-{name}")
        return self.cfg_path


def make_world(root: Path, monkeypatch, *, profile: str = "smoke", algorithms=None, build: bool = True,
               runs=None) -> World:
    """smoke: all four fixture datasets; pilot/confirmatory: SOP only (real-data rules allow no fixtures)."""
    data_root, out_root = root / "data", root / "out"
    if profile == "smoke":
        counts = make_all(data_root)
        d = smoke_cfg_dict(counts, data_root, out_root)
    else:
        counts = {"sop": make_sop(data_root / "sop")}
        d = smoke_cfg_dict({**counts, "kadid10k": {}, "hpatches": {}, "mvtec_ad": {}}, data_root, out_root)
        d["datasets"] = {"sop": {**d["datasets"]["sop"], "fixture": False}}
        d["profile"] = profile
        d["seeds"] = {"seed": 11, "seeds": [11, 23, 37] if profile == "pilot" else [11, 23, 37, 51, 71]}
    d["algorithms"] = dict(algorithms if algorithms is not None else {"fake": "1"})
    reg = root / "protocol_registry.json"
    register_version(reg, "1", None)
    world = World(root, d, root / "cfg.yaml", reg, out_root)
    world.runs = list(runs if runs is not None else [{"module": "photometric", "command": "run", "config": "cfg.yaml"}])
    world.write()
    monkeypatch.setattr(registry, "_PAIR_BUILDERS", {p: toy_pairs(p) for p in RELATION})
    if build:
        build_manifests(load_config(world.cfg_path, module=TEST_MODULE), data_root, Path(d["paths"]["manifest_dir"]))
    return world
