"""Verify a lock file by a fresh install (spec §12.2).

Usage: python scripts/verify_lock.py requirements.lock ["unit or smoke"]
Creates a clean virtual environment, installs the lock (`pip install --no-deps -r`), the project itself
(`pip install --no-deps -e .`), and runs `pytest -m <markers>`.
Exit codes: 0 OK, 1 install failed, 2 tests failed, 3 interpreter differs from the lock's Python line.
"""

import re
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _lock_python(lock: Path) -> str | None:
    m = re.search(r"^# Python (\d+\.\d+)", lock.read_text(encoding="utf-8"), re.M)
    return m[1] if m else None


def main(argv: list[str]) -> int:
    lock = Path(argv[0]).resolve()
    markers = argv[1] if len(argv) > 1 else "unit or smoke"
    wanted, current = _lock_python(lock), f"{sys.version_info.major}.{sys.version_info.minor}"
    if wanted != current:
        print(f"{lock.name} was verified on Python {wanted}, this is {current}", file=sys.stderr)
        return 3
    with tempfile.TemporaryDirectory() as tmp:
        env = Path(tmp) / "env"
        venv.create(env, with_pip=True)
        python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        for cmd in (["-m", "pip", "install", "-q", "--no-deps", "-r", str(lock)],
                    ["-m", "pip", "install", "-q", "--no-deps", "-e", str(REPO)]):
            if subprocess.run([str(python), *cmd], cwd=REPO).returncode != 0:
                print(f"install failed: {' '.join(cmd)}", file=sys.stderr)
                return 1
        tests = subprocess.run([str(python), "-m", "pytest", "-m", markers, "-q", "-p", "no:cacheprovider"], cwd=REPO)
        return 0 if tests.returncode == 0 else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
