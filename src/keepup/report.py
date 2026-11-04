"""charts and the html report.

one figure per table, three panels sharing the time axis: dead rows against
the vacuum threshold, table size against what the live rows need, and xid
age against the anti-wraparound and failsafe lines. separate panels instead
of twin y-axes on purpose; those numbers live on unrelated scales.
"""

from __future__ import annotations

import base64
import html
import io
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from .gucs import XID_STOP_LIMIT
from .profile import BLCKSZ, Profile
from .sim.results import MB, SimResult, TableSummary, _fmt_dur

# colours: categorical slots 1-3 of a palette that was checked for colour
# blindness, plus the reserved "critical" red for hard limits
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
CRITICAL = "#d03b3b"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


@dataclass
class Bands:
    """p10/p50/p90 across monte carlo runs, per table, for size and xid age."""

    size_mb: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    age: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    bloat_end: dict[str, tuple[float, float, float]]
    runs: int


def bands_from(results: list[SimResult]) -> Bands | None:
    if len(results) < 3:
        return None
    size: dict[str, Any] = {}
    age: dict[str, Any] = {}
    bend: dict[str, tuple[float, float, float]] = {}
    for name in results[0].tables:
        s = np.stack([r.tables[name].pages * BLCKSZ / MB for r in results])
        a = np.stack([r.tables[name].age for r in results])
        size[name] = tuple(np.quantile(s, q, axis=0) for q in (0.1, 0.5, 0.9))
        age[name] = tuple(np.quantile(a, q, axis=0) for q in (0.1, 0.5, 0.9))
        b = [float(r.tables[name].bloat[-1]) for r in results]
        q10, q50, q90 = (float(x) for x in np.quantile(b, [0.1, 0.5, 0.9]))
        bend[name] = (q10, q50, q90)
    return Bands(size, age, bend, len(results))


def _date_fmt() -> Any:
    return mdates.DateFormatter("%b %d")  # type: ignore[no-untyped-call]


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _label(ax: Any, x: Any, y: float, text: str, color: str = INK2) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(4, 3),
        textcoords="offset points",
        fontsize=7.5,
        color=color,
        va="bottom",
    )


def table_figure(res: SimResult, name: str, bands: Bands | None = None) -> bytes:
    ts = res.tables[name]
    t = [res.start + timedelta(seconds=float(s)) for s in res.sample_times]
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 6.6), sharex=True, dpi=110)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        _style(ax)

    # dead rows vs threshold
    ax = axes[0]
    ax.plot(t, ts.dead_phys, color=BLUE, linewidth=1.6, label="dead rows (on disk)")
    ax.plot(
        t,
        ts.dead_stats,
        color=ORANGE,
        linewidth=1.0,
        alpha=0.9,
        label="n_dead_tup (what autovacuum sees)",
    )
    ax.plot(
        t, ts.vac_threshold, color=MUTED, linewidth=1.0, linestyle="--", label="vacuum threshold"
    )
    ax.set_ylabel("rows", color=INK2, fontsize=8.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.EngFormatter())
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=3, labelcolor=INK2)
    vac = [r for r in res.runs_for(name) if r.vacuum]
    wrap = [r for r in vac if r.wraparound]
    if 0 < len(vac) <= 400:
        ymax = float(np.max(ts.dead_phys)) if len(ts.dead_phys) else 1.0
        xs = [res.start + timedelta(seconds=r.start) for r in vac if not r.wraparound]
        ax.scatter(xs, [ymax * 1.02] * len(xs), marker="|", s=40, color=INK2, linewidths=0.8)
        if wrap:
            xw = [res.start + timedelta(seconds=r.start) for r in wrap]
            ax.scatter(
                xw, [ymax * 1.02] * len(xw), marker="|", s=60, color=CRITICAL, linewidths=1.4
            )

    # size vs what live rows need
    ax = axes[1]
    size = ts.pages * BLCKSZ / MB
    ideal = ts.live / ts.tpp * BLCKSZ / MB
    if bands and name in bands.size_mb:
        lo, mid, hi = bands.size_mb[name]
        ax.fill_between(
            t, lo, hi, color=BLUE, alpha=0.14, linewidth=0, label=f"p10-p90 over {bands.runs} runs"
        )
        ax.plot(t, mid, color=BLUE, linewidth=1.6, label="table size (median)")
    else:
        ax.plot(t, size, color=BLUE, linewidth=1.6, label="table size")
    ax.plot(t, ideal, color=AQUA, linewidth=1.2, linestyle="-", label="size if packed")
    ax.set_ylabel("MB", color=INK2, fontsize=8.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.EngFormatter())
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=3, labelcolor=INK2)
    b = ts.bloat
    _label(ax, t[-1], float(size[-1]), f"{b[-1]:.0%} bloat", INK2)

    # xid age
    ax = axes[2]
    if bands and name in bands.age:
        lo, mid, hi = bands.age[name]
        ax.fill_between(t, lo, hi, color=BLUE, alpha=0.14, linewidth=0)
        ax.plot(t, mid, color=BLUE, linewidth=1.6, label="age(relfrozenxid)")
    else:
        ax.plot(t, ts.age, color=BLUE, linewidth=1.6, label="age(relfrozenxid)")
    ax.axhline(ts.freeze_max_age, color=MUTED, linestyle="--", linewidth=1.0)
    _label(ax, t[0], ts.freeze_max_age, "autovacuum_freeze_max_age")
    top = float(np.max(ts.age)) if len(ts.age) else 0.0
    if top > 0.5 * XID_STOP_LIMIT:
        ax.axhline(XID_STOP_LIMIT, color=CRITICAL, linewidth=1.2)
        _label(ax, t[0], XID_STOP_LIMIT, "writes stop here", CRITICAL)
    ax.set_ylabel("xids", color=INK2, fontsize=8.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.EngFormatter())
    ax.set_ylim(bottom=0, top=max(top, ts.freeze_max_age) * 1.15)
    ax.xaxis.set_major_formatter(_date_fmt())

    fig.suptitle(name, x=0.07, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE)
    plt.close(fig)
    return buf.getvalue()


def cluster_figure(res: SimResult) -> bytes:
    """worker occupancy and vacuum i/o over the horizon, hourly averages."""
    t5, busy = res.busy_series(300.0)
    _, rd, wr = res.io_series(300.0)
    # 5 minute buckets -> hourly means; a partial last hour is dropped
    step = 12 if len(busy) >= 12 else 1
    k = len(busy) // step * step

    def hourly(x: np.ndarray) -> np.ndarray:
        return x[:k].reshape(-1, step).mean(axis=1) if k else x

    th = [res.start + timedelta(seconds=float(x)) for x in t5[:k:step]]
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 4.4), sharex=True, dpi=110)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        _style(ax)
    ax = axes[0]
    ax.plot(th, hourly(busy), color=BLUE, linewidth=1.4)
    ax.axhline(res.settings.autovacuum_max_workers, color=MUTED, linestyle="--", linewidth=1.0)
    _label(ax, th[0], res.settings.autovacuum_max_workers, "autovacuum_max_workers")
    ax.set_ylabel("busy workers", color=INK2, fontsize=8.5)
    ax.set_ylim(0, res.settings.autovacuum_max_workers * 1.25)
    ax = axes[1]
    ax.plot(th, hourly(rd), color=BLUE, linewidth=1.4, label="read")
    ax.plot(th, hourly(wr), color=ORANGE, linewidth=1.4, label="dirtied")
    ax.set_ylabel("MB/s (hourly avg)", color=INK2, fontsize=8.5)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=2, labelcolor=INK2)
    ax.xaxis.set_major_formatter(_date_fmt())
    fig.suptitle("autovacuum, whole cluster", x=0.07, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE)
    plt.close(fig)
    return buf.getvalue()


def summary_rows(summaries: list[TableSummary], bands: Bands | None) -> list[dict[str, Any]]:
    rows = []
    for s in summaries:
        b = bands.bloat_end.get(s.name) if bands else None
        rows.append(
            {
                "table": s.name,
                "verdict": s.verdict,
                "size": f"{_size(s.size_mb_start)} -> {_size(s.size_mb_end)}",
                "bloat": (
                    f"{s.bloat_start:.0%} -> {s.bloat_end:.0%}"
                    + (f" (p90 {b[2]:.0%})" if b else "")
                ),
                "vacuums": f"{s.vacuums} ({s.wraparound} anti-wrap)"
                if s.wraparound
                else str(s.vacuums),
                "max xid age": f"{s.age_max / 1e6:,.0f}M",
                "p95 wait": _fmt_dur(s.lag_p95_s),
                "longest run": _fmt_dur(s.longest_run_s),
                "why": "; ".join(s.reasons),
            }
        )
    return rows


def _size(mb: float) -> str:
    if mb >= 10 * 1024:
        return f"{mb / 1024:,.0f} GB"
    return f"{mb:,.0f} MB"


def text_table(rows: list[dict[str, Any]], cols: list[str] | None = None) -> str:
    if not rows:
        return ""
    cols = cols or [c for c in rows[0] if c != "why"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    out = [line, "  ".join("-" * widths[c] for c in cols)]
    for r in rows:
        out.append("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    return "\n".join(out)


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode()


_CSS = """
:root { --ink: #0b0b0b; --ink2: #52514e; --muted: #898781; --line: #e1e0d9;
        --bg: #f9f9f7; --card: #fcfcfb; --ok: #0ca30c; --warn: #fab219; --bad: #d03b3b; }
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink);
       background: var(--bg); margin: 0; padding: 24px 16px; }
main { max-width: 980px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 32px 0 8px; }
p, li { color: var(--ink2); line-height: 1.5; }
.wrap { overflow-x: auto; }
table { border-collapse: collapse; font-size: 13px; width: 100%; background: var(--card); }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line);
         vertical-align: top; }
th { color: var(--ink2); font-weight: 600; }
td.num { font-variant-numeric: tabular-nums; }
.v { font-weight: 600; white-space: nowrap; }
.v::before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 4px;
             margin-right: 6px; vertical-align: middle; }
.v.ok::before { background: var(--ok); } .v.behind::before { background: var(--warn); }
.v.danger::before { background: var(--bad); }
img { max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 4px;
      background: var(--card); }
pre { background: var(--card); border: 1px solid var(--line); padding: 12px; overflow-x: auto;
      font-size: 12.5px; }
.small { font-size: 12px; color: var(--muted); }
"""


def render_figures(
    res: SimResult, summaries: list[TableSummary], bands: Bands | None = None
) -> dict[str, bytes]:
    figs = {s.name: table_figure(res, s.name, bands) for s in summaries}
    figs["_cluster"] = cluster_figure(res)
    return figs


def html_report(
    profile: Profile,
    res: SimResult,
    summaries: list[TableSummary],
    bands: Bands | None = None,
    patch_sql: str | None = None,
    title: str = "autovacuum forecast",
    notes: list[str] | None = None,
    figures: dict[str, bytes] | None = None,
) -> str:
    days = res.horizon / 86400
    figs = figures if figures is not None else render_figures(res, summaries, bands)
    rows = summary_rows(summaries, bands)
    worst = max((s.verdict for s in summaries), key=["ok", "behind", "danger"].index, default="ok")
    headline = {
        "ok": "keeps up on every table over the forecast window",
        "behind": "falls behind on at least one table",
        "danger": "heads toward xid wraparound trouble on at least one table",
    }[worst]
    parts = [
        f"<!doctype html><html lang=en><head><meta charset=utf-8>"
        f"<meta name=viewport content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body><main>"
    ]
    parts.append(f"<h1>{html.escape(title)}</h1>")
    parts.append(
        f"<p>{days:.0f} days from {res.start:%Y-%m-%d %H:%M} {res.start.tzname()}, "
        f"postgres {profile.settings.version}, {len(summaries)} tables. "
        f"Autovacuum {headline}.</p>"
    )
    parts.append(
        f"<p class=small>workers busy {res.utilization:.0%} of the time on average, "
        f"all {res.settings.autovacuum_max_workers} busy for "
        f"{res.saturated_fraction:.1%} of it. p99 vacuum i/o "
        f"{res.peak_io_mbps():.1f} MB/s.</p>"
    )
    for n in notes or []:
        parts.append(f"<p class=small>{html.escape(n)}</p>")
    parts.append("<div class=wrap><table><tr>")
    cols = [
        "table",
        "verdict",
        "size",
        "bloat",
        "vacuums",
        "max xid age",
        "p95 wait",
        "longest run",
    ]
    parts.extend(f"<th>{c}</th>" for c in cols)
    parts.append("</tr>")
    for r in rows:
        parts.append("<tr>")
        for c in cols:
            v = html.escape(str(r[c]))
            if c == "verdict":
                parts.append(f"<td><span class='v {v}'>{v}</span></td>")
            else:
                parts.append(f"<td class=num>{v}</td>")
        parts.append("</tr>")
        if r["why"]:
            parts.append(
                f"<tr><td></td><td colspan={len(cols) - 1} class=small>"
                f"{html.escape(r['why'])}</td></tr>"
            )
    parts.append("</table></div>")
    if patch_sql:
        parts.append("<h2>suggested settings</h2>")
        parts.append(f"<pre>{html.escape(patch_sql)}</pre>")
    parts.append("<h2>cluster</h2>")
    parts.append(
        f"<img alt='worker occupancy and vacuum io' "
        f"src='data:image/png;base64,{_b64(figs['_cluster'])}'>"
    )
    for s in summaries:
        parts.append(f"<h2>{html.escape(s.name)}</h2>")
        parts.append(
            f"<img alt='forecast for {html.escape(s.name)}' "
            f"src='data:image/png;base64,{_b64(figs[s.name])}'>"
        )
    parts.append(
        "<p class=small>dead rows on disk vs n_dead_tup: vacuum overwrites the "
        "stat with what it counted, so rows that died behind the scan go unseen "
        "until the next analyze. tick marks on the first panel are vacuum starts, "
        "red ones are anti-wraparound.</p>"
    )
    parts.append("</main></body></html>")
    return "".join(parts)


def write_outputs(
    out: Path,
    profile: Profile,
    res: SimResult,
    summaries: list[TableSummary],
    bands: Bands | None = None,
    patch_sql: str | None = None,
    title: str = "autovacuum forecast",
    notes: list[str] | None = None,
    pngs: bool = True,
) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    figs = render_figures(res, summaries, bands)
    p = out / "report.html"
    p.write_text(html_report(profile, res, summaries, bands, patch_sql, title, notes, figs))
    written.append(p)
    p = out / "summary.json"
    p.write_text(
        json.dumps(
            {
                "start": res.start.isoformat(),
                "days": res.horizon / 86400,
                "utilization": res.utilization,
                "saturated_fraction": res.saturated_fraction,
                "peak_io_mbps": res.peak_io_mbps(),
                "tables": [s.to_dict() for s in summaries],
                "bands": None
                if bands is None
                else {
                    k: {"p10": v[0], "p50": v[1], "p90": v[2]} for k, v in bands.bloat_end.items()
                },
            },
            indent=2,
        )
        + "\n"
    )
    written.append(p)
    if pngs:
        charts = out / "charts"
        charts.mkdir(exist_ok=True)
        for s in summaries:
            safe = s.name.replace("/", "_")
            q = charts / f"{safe}.png"
            q.write_bytes(figs[s.name])
            written.append(q)
        q = charts / "_cluster.png"
        q.write_bytes(figs["_cluster"])
        written.append(q)
    if patch_sql:
        p = out / "patch.sql"
        p.write_text(patch_sql)
        written.append(p)
    return written
