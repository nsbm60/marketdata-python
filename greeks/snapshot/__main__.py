"""`python -m greeks.snapshot ...` entry point."""

from __future__ import annotations

import sys

from greeks.snapshot.run import main

if __name__ == "__main__":
    sys.exit(main())
