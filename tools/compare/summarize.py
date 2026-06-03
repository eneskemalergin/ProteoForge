#!/usr/bin/env python3
"""Summarize zebrac reference vs ProteoForge results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ms(entry: dict[str, Any]) -> float:
    return entry["wall_time"]["median"] / 1_000_000.0


def _rss_mb(entry: dict[str, Any]) -> float:
    return entry["peak_rss"]["median"] / (1024 * 1024)


def _row(label: str, ref: dict[str, Any], pf: dict[str, Any]) -> dict[str, Any]:
    ref_ms, pf_ms = _ms(ref), _ms(pf)
    ref_rss, pf_rss = _rss_mb(ref), _rss_mb(pf)
    speedup = (1.0 - pf_ms / ref_ms) * 100.0 if ref_ms else 0.0
    rss_delta = ((pf_rss - ref_rss) / ref_rss) * 100.0 if ref_rss else 0.0
    return {
        "label": label,
        "reference_wall_ms": round(ref_ms, 1),
        "proteoforge_wall_ms": round(pf_ms, 1),
        "speedup_pct": round(speedup, 1),
        "reference_rss_mb": round(ref_rss, 1),
        "proteoforge_rss_mb": round(pf_rss, 1),
        "rss_delta_pct": round(rss_delta, 1),
    }


def summarize_json(path: Path, *, label: str | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    if len(results) < 2:
        msg = f"Expected 2 zebrac results in {path}"
        raise ValueError(msg)
    case_label = label or path.parent.name
    return _row(case_label, results[0], results[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_files",
        nargs="*",
        type=Path,
        help="zebrac JSON files (reference first, proteoforge second inside each).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Directory with <case>/zebrac.json layout.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        help="Case names under --work-dir (e.g. complete medium large).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write summary JSON.",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    if args.work_dir is not None:
        if not args.cases:
            parser.error("--work-dir requires --cases")
        for case in args.cases:
            json_path = args.work_dir / case / "zebrac.json"
            if json_path.is_file():
                rows.append(summarize_json(json_path, label=case))
            else:
                print(f"warning: missing {json_path}", file=sys.stderr)
    else:
        for json_path in args.json_files:
            rows.append(summarize_json(json_path))

    if not rows:
        print("No results to summarize.")
        return

    headers = (
        "case",
        "ref_ms",
        "pf_ms",
        "speedup%",
        "ref_rss",
        "pf_rss",
        "rss+%",
    )
    print("\t".join(headers))
    for row in rows:
        print(
            "\t".join(
                [
                    row["label"],
                    str(row["reference_wall_ms"]),
                    str(row["proteoforge_wall_ms"]),
                    str(row["speedup_pct"]),
                    str(row["reference_rss_mb"]),
                    str(row["proteoforge_rss_mb"]),
                    str(row["rss_delta_pct"]),
                ]
            )
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"cases": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
