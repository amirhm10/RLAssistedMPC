"""Reuse the previous read-only scientific guards with fresh resource samples."""
import hashlib
import json
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "run_audit_20261022_1340/inspect_live.py"
audit = runpy.run_path(str(SOURCE))["audit"]
audit.__globals__["HERE"] = HERE
audit(HERE / "audit.json")
with (HERE / "audit_source.json").open("x", encoding="utf-8") as stream:
    json.dump(dict(path=str(SOURCE), sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest()), stream, indent=2)
    stream.write("\n")
