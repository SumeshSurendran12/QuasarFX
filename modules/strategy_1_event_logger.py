from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except Exception:  # pragma: no cover - py<3.9 fallback environments
    ZoneInfo = None  # type: ignore[assignment]
    ZoneInfoNotFoundError = Exception  # type: ignore[assignment]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _local_date_iso(timezone_name: str) -> str:
    if ZoneInfo is None:
        return datetime.now().date().isoformat()
    try:
        return datetime.now(ZoneInfo(timezone_name)).date().isoformat()
    except ZoneInfoNotFoundError:
        return datetime.now().date().isoformat()


def _canonical_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class Strategy1CanonicalEventLogger:
    def __init__(
        self,
        *,
        manifest_path: Optional[Path] = None,
        profile_path: Optional[Path] = None,
        events_path: Optional[Path] = None,
        session_label: str = "LONDON",
        mode: str = "paper",
        symbol: str = "EURUSD",
        broker: str = "gcapi_demo",
        timezone_name: str = "America/Chicago",
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        self.root = root
        self.manifest_path = manifest_path or (root / "manifest.json")
        self.profile_path = profile_path or (root / "strategy_1_profile.json")

        manifest: Dict[str, Any] = {}
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))

        profile: Dict[str, Any] = {}
        if self.profile_path.exists():
            profile = json.loads(self.profile_path.read_text(encoding="utf-8"))

        event_contract = manifest.get("event_contract", {}) if isinstance(manifest.get("event_contract"), dict) else {}
        profile_event_contract = profile.get("event_contract", {}) if isinstance(profile.get("event_contract"), dict) else {}

        events_file = str(event_contract.get("events_file", "")).strip()
        if not events_file:
            events_file = str((profile.get("reporting", {}) or {}).get("events_file", "")).strip()
        if not events_file:
            events_file = "events.jsonl"

        if events_path is None:
            self.events_path = root / events_file
        else:
            self.events_path = events_path
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

        self.schema_version = str(manifest.get("schema_version") or profile.get("schema_version") or "1.0.0").strip() or "1.0.0"
        self.manifest_version = str(manifest.get("manifest_version") or profile.get("manifest_version") or "1.0.0").strip() or "1.0.0"
        self.stage = str(manifest.get("stage") or profile.get("stage") or "PAPER_CANDIDATE").strip() or "PAPER_CANDIDATE"
        self.strategy_id = str(manifest.get("strategy_id") or "strategy_1").strip() or "strategy_1"
        self.profile_name = str(profile.get("profile_id") or manifest.get("manifest_id") or "strategy_1_profile").strip()
        self.timezone_name = timezone_name
        self.symbol = symbol
        self.mode = mode
        self.broker = broker
        self.session_label = "".join(ch for ch in str(session_label).upper() if ch.isalnum()) or "LONDON"
        self.process_start_ts = _utc_now_iso()
        self.reason_codes: Set[str] = {
            str(x)
            for x in (
                event_contract.get("reason_codes")
                or profile_event_contract.get("reason_codes")
                or [
                    "signal_pass",
                    "spread_gate",
                    "session_cap",
                    "daily_loss_limit",
                    "max_open_positions",
                    "outside_session",
                    "cooldown_active",
                    "duplicate_signal",
                    "no_liquidity",
                    "manual_disable",
                    "policy_breach",
                    "within_limits",
                    "broker_api_failure",
                ]
            )
        }

        profile_hash = ""
        if self.manifest_path.exists():
            profile_hash = f"sha256:{_canonical_sha256(self.manifest_path)}"
        elif self.profile_path.exists():
            profile_hash = f"sha256:{_canonical_sha256(self.profile_path)}"
        if not profile_hash:
            profile_hash = f"sha256:{'0' * 64}"
        self.profile_hash = profile_hash

        self.run_id = self._generate_unique_run_id()

        controls = manifest.get("controls", {}) if isinstance(manifest.get("controls"), dict) else {}
        self.max_trades_per_session = int(controls.get("max_trades_per_session", 1))
        if self.max_trades_per_session <= 0:
            self.max_trades_per_session = 1

    def _run_id_exists(self, run_id: str) -> bool:
        if not self.events_path.exists():
            return False
        needle = f"\"run_id\":\"{run_id}\""
        with self.events_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if needle in line:
                    return True
        return False

    def _generate_run_id(self, session_label: str) -> str:
        script_path = self.root / "scripts" / "generate_strategy_1_run_id.py"
        if script_path.exists():
            cmd = [
                sys.executable,
                str(script_path),
                "--manifest",
                str(self.manifest_path),
                "--timezone",
                self.timezone_name,
                "--session",
                session_label,
                "--profile-hash",
                self.profile_hash,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(self.root))
            candidate = str(result.stdout or "").strip()
            if result.returncode == 0 and candidate:
                return candidate

        run_day = _local_date_iso(self.timezone_name)
        return f"{run_day}_{session_label}_sha{self.profile_hash.replace('sha256:', '')[:8]}"

    def _generate_unique_run_id(self) -> str:
        seed = datetime.now(timezone.utc).strftime("%H%M%S%f")[:9]
        base_session = f"{self.session_label}{seed}"
        for attempt in range(1000):
            session_value = base_session if attempt == 0 else f"{base_session}{attempt:02d}"
            candidate = self._generate_run_id(session_value)
            if not self._run_id_exists(candidate):
                return candidate
        raise RuntimeError("Unable to generate unique run_id after 1000 attempts.")

    def append(self, event: str, **fields: Any) -> Dict[str, Any]:
        rec: Dict[str, Any] = {
            "ts": _utc_now_iso(),
            "run_id": self.run_id,
            "event": str(event),
            "stage": self.stage,
            "strategy_id": self.strategy_id,
            "profile_hash": self.profile_hash,
            "schema_version": self.schema_version,
            "manifest_version": self.manifest_version,
            "process_start_ts": self.process_start_ts,
        }
        rec.update(fields)

        reason_code = str(rec.get("reason_code", "")).strip()
        if reason_code and self.reason_codes and reason_code not in self.reason_codes:
            raise ValueError(f"Invalid reason_code '{reason_code}' not in canonical reason code set.")

        line = json.dumps(rec, separators=(",", ":"), ensure_ascii=False)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return rec
