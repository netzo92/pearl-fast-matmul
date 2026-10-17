#!/usr/bin/env python3
"""Recover a checksummed benchmark report from short GCP serial-log chunks."""
import argparse
import base64
import gzip
import hashlib
import json
import re
from pathlib import Path


def decode(text):
    hashes = set(re.findall(r"PEARL_REPORT_SHA256=([0-9a-f]{64})", text))
    if len(hashes) != 1:
        raise ValueError("expected one report SHA256")
    chunks, totals = {}, set()
    for index, total, value in re.findall(r"PEARL_REPORT_CHUNK=(\d+)/(\d+):([A-Za-z0-9+/=]+)", text):
        index, total = int(index), int(total)
        totals.add(total)
        if index in chunks and chunks[index] != value:
            raise ValueError("conflicting duplicate chunk")
        chunks[index] = value
    if len(totals) != 1:
        raise ValueError("missing chunks or conflicting chunk counts")
    count = totals.pop()
    if count <= 0 or set(chunks) != set(range(count)):
        raise ValueError("incomplete report")
    compressed = base64.b64decode("".join(chunks[i] for i in range(count)), validate=True)
    raw = gzip.decompress(compressed)
    if hashlib.sha256(raw).hexdigest() not in hashes:
        raise ValueError("report SHA256 mismatch")
    json.loads(raw)
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial_log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = decode(args.serial_log.read_text(errors="replace"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(f"Verified {hashlib.sha256(raw).hexdigest()}; saved {args.output}")


if __name__ == "__main__":
    main()
