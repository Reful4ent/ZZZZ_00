"""Generate JSON Schema files of the data contract from the code (spec §12.1 п.4).

Usage: python scripts/gen_schemas.py [out_dir]   (default: schemas/)
A test checks that the committed schemas/ equal a fresh generation.
"""

import dataclasses
import json
import sys
from pathlib import Path

from expcore.schemas import (DATASET_FIELDS, DATASET_OPTIONAL, IMAGES_COLUMNS, IMAGES_NULLABLE, PAIRS_COLUMNS,
                             PAIRS_NULLABLE, PREDICTIONS_COLUMNS, PREDICTIONS_NULLABLE, RUN_FIELDS, RUN_OPTIONAL,
                             COMPONENT_PREFIX, record_json_schema)
from expcore.thresholds import Threshold
from experiments.aggregate import METRICS_SCHEMA

DRAFT = "https://json-schema.org/draft/2020-12/schema"
JSON_TYPES = {"str": "string", "bool": "boolean", "int8": "integer", "int32": "integer", "int64": "integer",
              "float64": "number"}


def _typed(dtype: str, nullable: bool) -> dict:
    t = JSON_TYPES[dtype]
    return {"type": [t, "null"] if nullable else t, "x-dtype": dtype}


def table_schema(title: str, columns: dict, nullable, *, extra_pattern: str | None = None) -> dict:
    """One parquet row as a JSON object: every column present, dtype and nullability from the code."""
    schema = {"$schema": DRAFT, "title": title, "type": "object", "required": list(columns),
              "properties": {c: _typed(d, c in nullable) for c, d in columns.items()}, "additionalProperties": False}
    if extra_pattern:
        schema["patternProperties"] = {extra_pattern: _typed("float64", True)}
    return schema


def _thresholds() -> dict:
    py = {str: "str", float: "float64", int: "int64"}
    columns = {f.name: py[f.type] for f in dataclasses.fields(Threshold)}
    schema = table_schema("thresholds.json", columns, ())
    schema["required"].append("created_at")
    schema["properties"]["created_at"] = {"type": "string"}
    return schema


def schemas() -> dict[str, dict]:
    metrics_nullable = {c for c, d in METRICS_SCHEMA.items() if d in ("int64", "float64")} - {"seed", "n_pairs",
                                                                                              "n_source_groups", "n_seeds"}
    return {
        "datasets": {"$schema": DRAFT, "title": "datasets.json", "type": "array",
                     "items": record_json_schema(DATASET_FIELDS, DATASET_OPTIONAL)},
        "images": table_schema("images.parquet row", IMAGES_COLUMNS, IMAGES_NULLABLE),
        "pairs": table_schema("pairs.parquet row", PAIRS_COLUMNS, PAIRS_NULLABLE),
        "predictions": table_schema("predictions.parquet row", PREDICTIONS_COLUMNS, PREDICTIONS_NULLABLE,
                                    extra_pattern=f"^{COMPONENT_PREFIX}"),
        "run": {"$schema": DRAFT, "title": "run.json", **record_json_schema(RUN_FIELDS, RUN_OPTIONAL)},
        "thresholds": _thresholds(),
        "metrics": table_schema("metrics.parquet row", METRICS_SCHEMA, metrics_nullable),
    }


def generate(out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, schema in schemas().items():
        path = out_dir / f"{name}.schema.json"
        text = json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        path.write_bytes(text.encode("utf-8"))   # LF on every OS: the committed files are compared byte-wise
        paths.append(path)
    return paths


if __name__ == "__main__":
    for p in generate(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "schemas"):
        print(p)
