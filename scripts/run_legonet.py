"""Thin subprocess entry point used by the Streamlit GUI.

The GUI builds a CLI-compatible command and launches this script so experiments
run separately from the Streamlit process. Command-line users should normally
use the installed ``legonet`` command; PyCharm and other IDE users who want to
edit run settings in code should use ``scripts/debug_legonet.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from legonet.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
