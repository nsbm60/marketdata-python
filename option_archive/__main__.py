"""Entry point.

  python -m option_archive          — run the single backfill program (the drain)
  python -m option_archive probe …  — characterize a Massive outage (diagnostic)

Backfill logic lives in ``option_archive.archive``; the probe in
``option_archive.probe``.
"""

import sys

from option_archive.archive import main as archive_main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        from option_archive.probe import main as probe_main

        probe_main(sys.argv[2:])
    else:
        archive_main()
