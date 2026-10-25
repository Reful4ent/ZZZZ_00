"""CLI: python -m photometric run --config configs/p1_confirm.yaml"""

from __future__ import annotations

import argparse


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m photometric", description="P1 photometric (ТЗ 01)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="полный прогон опытов P1-E1..E4")
    r.add_argument("--config", required=True)
    r.add_argument("--output-dir", help="переопределить output_dir конфигурации")
    v = sub.add_parser("check-data", help="только проверить состав KADID-10k")
    v.add_argument("--config", required=True)
    args = ap.parse_args(argv)

    from photometric.config import load_config, resolve_path

    cfg = load_config(args.config)
    if args.cmd == "check-data":
        from photometric.kadid import load_kadid

        k = cfg["kadid"]
        images = load_kadid(resolve_path(cfg, k["root"]), k["images_subdir"], k["dmos_csv"], validate_full=True)
        print(f"KADID-10k OK: {int((images['type_id'] == '00').sum())} originals, {int((images['type_id'] != '00').sum())} distorted")
        return 0
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    from photometric.pipeline import run

    out = run(cfg)
    print(out)
    return 0
