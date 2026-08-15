"""Make the repo root importable so integration tests can use `discovery`, `service`, etc.

(Only runs under pytest; the test scripts also self-insert the path so they run standalone.)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
