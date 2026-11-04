"""regenerate docs/img/before-after.png for the readme.

    python scripts/readme_charts.py examples/upgraded.toml \
        --set autovacuum_vacuum_cost_delay=0.5 ...

the --set / --table-set values are the ones `keepup tune` suggested, pasted
in by hand, so this doesn't rerun the search every time.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from keepup.cli import _apply_overrides
from keepup.profile import BLCKSZ, load_profile
from keepup.report import AXIS, BLUE, GRID, INK, INK2, MUTED, ORANGE, SURFACE, _date_fmt
from keepup.sim import simulate
from keepup.sim.results import MB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--table-set", action="append", default=[])
    ap.add_argument("--days", type=float, default=30)
    ap.add_argument("--age-table", default="public.events")
    ap.add_argument("--size-table", default="public.inventory")
    ap.add_argument("--out", default="docs/img/before-after.png")
    a = ap.parse_args()

    base = load_profile(a.profile)
    new = _apply_overrides(base, a.set, a.table_set)
    before = simulate(base, days=a.days)
    after = simulate(new, days=a.days)
    t = [before.start + timedelta(seconds=float(s)) for s in before.sample_times]

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.4), sharex=True, dpi=130)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(matplotlib.ticker.EngFormatter())

    ax = axes[0]
    tb, ta = before.tables[a.age_table], after.tables[a.age_table]
    ax.plot(t, tb.age, color=ORANGE, linewidth=1.5, label="current settings")
    ax.plot(t, ta.age, color=BLUE, linewidth=1.5, label="after the patch")
    ax.axhline(tb.freeze_max_age, color=MUTED, linestyle="--", linewidth=1.0)
    ax.annotate(
        "autovacuum_freeze_max_age",
        (t[0], tb.freeze_max_age),
        xytext=(4, 3),
        textcoords="offset points",
        fontsize=7.5,
        color=INK2,
    )
    ax.set_ylim(0, max(float(np.max(tb.age)), tb.freeze_max_age) * 1.15)
    ax.set_title(f"{a.age_table}: age(relfrozenxid)", loc="left", fontsize=9.5, color=INK)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False, labelcolor=INK2)

    ax = axes[1]
    sb, sa = before.tables[a.size_table], after.tables[a.size_table]
    ax.plot(t, sb.pages * BLCKSZ / MB, color=ORANGE, linewidth=1.5, label="current settings")
    ax.plot(t, sa.pages * BLCKSZ / MB, color=BLUE, linewidth=1.5, label="after the patch")
    ax.plot(t, sb.live / sb.tpp * BLCKSZ / MB, color=MUTED, linewidth=1.0, linestyle="--")
    ax.annotate(
        "if packed",
        (t[-1], float(sb.live[-1] / sb.tpp * BLCKSZ / MB)),
        xytext=(-40, 4),
        textcoords="offset points",
        fontsize=7.5,
        color=INK2,
    )
    ax.set_ylim(bottom=0)
    ax.set_ylabel("MB", color=INK2, fontsize=8.5)
    ax.set_title(f"{a.size_table}: table size", loc="left", fontsize=9.5, color=INK)
    ax.xaxis.set_major_formatter(_date_fmt())

    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor=SURFACE)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
