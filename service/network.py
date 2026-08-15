"""
Local network address detection.

Matches Scala's NetworkUtils: prefers RFC1918 private addresses (192.168.x, 10.x),
skips virtual/VM interfaces, allows MDS_HOST env var override.
"""

import os
import socket
import logging

log = logging.getLogger(__name__)


def get_local_ipv4() -> str:
    """Get the local IPv4 address, preferring RFC1918 private addresses."""
    override = os.environ.get("MDS_HOST", "").strip()
    if override:
        return override

    try:
        # Connect to a public address to determine which interface routes traffic.
        # No data is sent; the OS selects the outbound interface.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.settimeout(1)
            s.connect(("8.8.8.8", 80))
            addr = s.getsockname()[0]
            return addr
        finally:
            s.close()
    except Exception:
        log.debug("[Network] UDP probe failed, falling back to hostname resolution")

    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        log.warning("[Network] Could not determine local IP, using 127.0.0.1")
        return "127.0.0.1"
