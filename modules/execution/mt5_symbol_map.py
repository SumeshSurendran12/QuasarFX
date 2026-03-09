from __future__ import annotations

from typing import Dict, Iterable, List


SYMBOL_MAP: Dict[str, List[str]] = {
    "EURUSD": ["EURUSD", "EURUSD.a", "EURUSDm", "EURUSD.r"],
    "GBPUSD": ["GBPUSD", "GBPUSD.a", "GBPUSDm", "GBPUSD.r"],
    "USDJPY": ["USDJPY", "USDJPY.a", "USDJPYm", "USDJPY.r"],
    "AUDUSD": ["AUDUSD", "AUDUSD.a", "AUDUSDm", "AUDUSD.r"],
    "USDCHF": ["USDCHF", "USDCHF.a", "USDCHFm", "USDCHF.r"],
    "USDCAD": ["USDCAD", "USDCAD.a", "USDCADm", "USDCAD.r"],
    "NZDUSD": ["NZDUSD", "NZDUSD.a", "NZDUSDm", "NZDUSD.r"],
}


def normalize_symbol(symbol: str) -> str:
    return str(symbol or "").upper().replace("/", "").strip()


def symbol_candidates(symbol: str) -> List[str]:
    normalized = normalize_symbol(symbol)
    out: List[str] = []
    for candidate in [normalized, *SYMBOL_MAP.get(normalized, [])]:
        if candidate and candidate not in out:
            out.append(candidate)
    return out or [normalized]


def resolve_symbol(symbol: str, available_symbols: Iterable[str]) -> str:
    available = [str(x) for x in available_symbols if str(x).strip()]
    if not available:
        return normalize_symbol(symbol)

    lower_map = {name.lower(): name for name in available}
    for candidate in symbol_candidates(symbol):
        if candidate in available:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    normalized = normalize_symbol(symbol)
    if normalized.lower() in lower_map:
        return lower_map[normalized.lower()]
    return normalized
