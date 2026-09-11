#!/usr/bin/env python3
"""Leakage tests for target_encoding.py.

These are the checks that decide whether the encodings are trustworthy. They use tiny
hand-built frames so every expected value can be computed by hand, and they run in
milliseconds, so there is no excuse for not running them before every experiment.

Usage:
    .venv/bin/python scripts/test_target_encoding.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import target_encoding as TE  # noqa: E402


def frame(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        {"month": [r[0] for r in rows], "AIRPORT": [r[1] for r in rows],
         "STAND_mvt": [r[2] for r in rows], "RUNWAY_mvt": [r[3] for r in rows],
         "hour": [r[4] for r in rows], "AIRCRAFT_OPERATOR_flt": [r[5] for r in rows],
         "ADES_mvt": [r[6] for r in rows], TE.TARGET: [r[7] for r in rows]},
        schema_overrides={TE.TARGET: pl.Int32},
    )


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -> ' + detail) if detail and not ok else ''}")
    return ok


def main() -> int:
    results = []

    # 1. A row's own target must not reach its own encoding.
    #    One stand, twelve months, eleven months at 600 s and month 1 at an extreme value.
    #    Month 1's encoding must be computed from the eleven 600 s rows alone.
    rows = [(1, "A", "S1", "R1", 8, "OP1", "D1", 50_000)]
    rows += [(m, "A", "S1", "R1", 8, "OP1", "D1", 600) for m in range(2, 13)]
    df = frame(rows)
    enc = TE.cross_fit_train(df)
    jan = enc.filter(pl.col("month") == 1)["te_stand"][0]
    others = df.filter(pl.col("month") != 1)
    expected = TE.fit_map(others, "te_stand", ["AIRPORT", "STAND_mvt"])["te_stand"][0]
    results.append(check("own target never enters own encoding",
                         abs(jan - expected) < 1e-9, f"{jan} vs {expected}"))
    results.append(check("the extreme row is encoded near the normal level, not near itself",
                         jan < 700, f"got {jan}"))

    # 2. Every block's encoding must equal a map fitted on exactly the complement.
    rows = [(m, "A", f"S{m % 3}", "R1", m % 24, "OP1", "D1", 500 + 40 * m) for m in range(1, 13)]
    rows += [(m, "B", f"S{m % 2}", "R2", m % 24, "OP2", "D2", 900 + 10 * m) for m in range(1, 13)]
    df = frame(rows)
    enc = TE.cross_fit_train(df)
    agree = True
    for b in range(1, 13):
        want = TE.transform(df.filter(pl.col("month") == b),
                            TE.fit_maps(df.filter(pl.col("month") != b)))
        got = enc.filter(pl.col("month") == b).select(want.columns)
        if not want.equals(got):
            agree = False
    results.append(check("cross fitting equals a map built on the complement, for all 12 blocks", agree))

    # 3. A group that appears only in the held-out block has no source rows, so it must be null.
    rows = [(1, "A", "ONLY_JAN", "R1", 8, "OP1", "D1", 700)]
    rows += [(m, "A", "S1", "R1", 8, "OP1", "D1", 700) for m in range(2, 13)]
    enc = TE.cross_fit_train(frame(rows))
    results.append(check("a group unseen in the source is encoded as null",
                         enc.filter(pl.col("STAND_mvt") == "ONLY_JAN")["te_stand"][0] is None))

    # 4. Winsorisation must cap a monster's contribution at CAP.
    base = [(m, "A", "S1", "R1", 8, "OP1", "D1", 600) for m in range(2, 13)]
    mild = TE.fit_map(frame(base + [(2, "A", "S1", "R1", 8, "OP1", "D1", 3600)]),
                      "te_stand", ["AIRPORT", "STAND_mvt"])["te_stand"][0]
    wild = TE.fit_map(frame(base + [(2, "A", "S1", "R1", 8, "OP1", "D1", 131_167)]),
                      "te_stand", ["AIRPORT", "STAND_mvt"])["te_stand"][0]
    results.append(check("a 131,167 s outlier moves the encoding no more than a 3,600 s one",
                         abs(mild - wild) < 1e-9, f"{mild} vs {wild}"))

    # 5. Smoothing must pull a thinly observed group toward its airport prior.
    rows = [(m, "A", "BUSY", "R1", 8, "OP1", "D1", 600) for m in range(1, 13) for _ in range(60)]
    rows += [(2, "A", "RARE", "R2", 9, "OP1", "D1", 3000)]
    df = frame(rows)
    m = TE.fit_map(df, "te_stand", ["AIRPORT", "STAND_mvt"])
    rare = m.filter(pl.col("STAND_mvt") == "RARE")["te_stand"][0]
    busy = m.filter(pl.col("STAND_mvt") == "BUSY")["te_stand"][0]
    results.append(check("a single observation is shrunk toward the airport prior",
                         abs(rare - 600) < 60, f"rare={rare:.1f}"))
    results.append(check("a heavily observed group keeps its own mean",
                         abs(busy - 600) < 1, f"busy={busy:.1f}"))

    # 6. The count column must reflect source support, not the row's own block.
    rows = [(m, "A", "S1", "R1", 8, "OP1", "D1", 600) for m in range(1, 13)]
    enc = TE.cross_fit_train(frame(rows))
    results.append(check("count excludes the held-out block (11, not 12)",
                         set(enc["te_stand_count"].to_list()) == {11},
                         str(set(enc["te_stand_count"].to_list()))))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
