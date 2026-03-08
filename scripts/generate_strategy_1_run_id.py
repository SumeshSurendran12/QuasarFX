#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ROOT = Path(__file__).resolve().parents[1]


def canonical_json_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate immutable Strategy 1 run_id: YYYY-MM-DD_SESSION_shaXXXXXXXX")
    p.add_argument("--manifest", default="manifest.json")
    p.add_argument("--date", default="", help="Trading date in YYYY-MM-DD. Defaults to now in --timezone.")
    p.add_argument("--timezone", default="America/Chicago")
    p.add_argument("--session", default="LONDON")
    p.add_argument("--profile-hash", default="", help="Optional full profile hash. Defaults to manifest sha256.")
    p.add_argument("--hash-prefix-len", type=int, default=8)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    session = "".join(ch for ch in str(args.session).upper() if ch.isalnum())
    if not session:
        raise ValueError("session must contain at least one alphanumeric character")

    if str(args.date).strip():
        trading_date = str(args.date).strip()
    else:
        tz_name = str(args.timezone).strip() or "America/Chicago"
        try:
            tz = ZoneInfo(tz_name)
            trading_date = datetime.now(tz).date().isoformat()
        except ZoneInfoNotFoundError:
            trading_date = datetime.now().date().isoformat()

    if str(args.profile_hash).strip():
        raw_hash = str(args.profile_hash).strip().lower().replace("sha256:", "")
    else:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        raw_hash = canonical_json_sha256(manifest_path)

    prefix_len = max(4, int(args.hash_prefix_len))
    run_id = f"{trading_date}_{session}_sha{raw_hash[:prefix_len]}"
    print(run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
