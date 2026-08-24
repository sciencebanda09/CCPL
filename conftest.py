"""Keep legacy flat module imports working for the historical test suite."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent
for source in (root / "ccpl", root / "ccpl" / "algorithms", root / "ccpl" / "environments"):
    source = str(source)
    if source not in sys.path:
        sys.path.insert(0, source)
