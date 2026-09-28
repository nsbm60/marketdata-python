"""Entry point: ``python -m option_archive`` runs the single backfill program.

All logic lives in ``option_archive.archive``; this is just the module hook so the
systemd unit (and a manual run) invoke the same one program.
"""

from option_archive.archive import main

if __name__ == "__main__":
    main()
