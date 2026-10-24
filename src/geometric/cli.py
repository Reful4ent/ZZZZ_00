"""python -m geometric run --config … — skeleton: full plumbing, the algorithm comes from the registry (exit 6 if none)."""

from dataclasses import dataclass
from typing import Sequence

from expcore.cli_common import command_parser, execute_run, run_cli
from expcore.config import register_params_schema

MODULE = "geometric"


@dataclass(frozen=True)
class GeometricParams:
    """Module parameters; fields are added by the program's TZ together with its algorithms."""


register_params_schema(MODULE, GeometricParams)


def main(argv: Sequence[str] | None = None) -> int:
    parser = command_parser(MODULE, ["run"])
    return run_cli(lambda args: execute_run(MODULE, args.command, args, query=False), argv, parser=parser)
