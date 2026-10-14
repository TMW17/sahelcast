"""Build the static website (site/index.html) from the serving tables.

The page is plain HTML plus Plotly (loaded from a CDN), so GitHub Pages can host it for free.
Charts are drawn in the light palette and re-coloured in the browser for dark mode.
Every chart has a table next to it, so no number is only readable by hovering.
"""

from __future__ import annotations

import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs_version

from .config import Settings
from .publish import read_serving, serving_dir

# Light values; the page swaps each one for its dark twin (see THEME_JS).
C = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "land": "#f0efec",
    "s1": "#2a78d6",  # forecast (categorical slot 1)
    "s2": "#eb6834",  # observed (categorical slot 2)
    "band90": "rgba(42,120,214,0.10)",
    "band50": "rgba(42,120,214,0.22)",
}
DARK = {
    "#fcfcfb": "#1a1a19",
    "#0b0b0b": "#ffffff",
    "#52514e": "#c3c2b7",
    "#e1e0d9": "#2c2c2a",
    "#c3c2b7": "#383835",
    "#f0efec": "#2c2c2a",
    "#2a78d6": "#3987e5",
    "#eb6834": "#d95926",
    "rgba(42,120,214,0.10)": "rgba(57,135,229,0.16)",
    "rgba(42,120,214,0.22)": "rgba(57,135,229,0.30)",
}
LOCAL = pd.Timedelta(hours=1)  # Algeria is UTC+1 all year
MODEL_NAMES = {
    "lightgbm": "LightGBM",
    "physical": "Physical",
    "persistence": "Persistence",
    "chronos2": "Chronos-2",
}
POLICY_NAMES = {
    "stochastic_calibrated": "Scenarios, calibrated",
    "stochastic_raw": "Scenarios, raw",
    "deterministic_calibrated_median": "Median, calibrated",
    "deterministic_raw_median": "Median, raw",
    "perfect_foresight": "Perfect foresight",
}
RECON_NAMES = {
    "level": "Level",
    "base (incoherent)": "Separate forecasts",
    "BottomUp": "Bottom-up",
    "MinTrace_method-wls_struct": "MinT (structural)",
    "MinTrace_method-mint_shrink": "MinT (shrinkage)",
}


def pretty(model: str) -> str:
    base = model.removesuffix("+cqr")
    name = MODEL_NAMES.get(base, base)
    return name + (" (calibrated)" if model.endswith("+cqr") else "")


def _layout(fig: go.Figure, height: int, **kwargs) -> go.Figure:
    axis = dict(
        gridcolor=C["grid"],
        gridwidth=1,
        linecolor=C["axis"],
        zeroline=False,
        tickfont=dict(color=C["muted"]),
        title_font=dict(color=C["ink2"]),
    )
    fig.update_layout(
        height=height,
        margin=dict(l=56, r=24, t=16, b=48),
        paper_bgcolor=C["surface"],
        plot_bgcolor=C["surface"],
        font=dict(
            family='system-ui, -apple-system, "Segoe UI", sans-serif', color=C["ink"], size=13
        ),
        hoverlabel=dict(bgcolor=C["surface"], bordercolor=C["axis"], font=dict(color=C["ink"])),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=C["ink2"])),
        xaxis=axis,
        yaxis=axis,
        template="none",
        **kwargs,
    )
    return fig


def fig_fan(fc: pd.DataFrame, title_units: str = "MW") -> go.Figure:
    fc = fc.sort_values("valid_time")
    x = fc["valid_time"] + LOCAL
    mw = lambda c: fc[c] * fc["ac_mw"]  # noqa: E731
    fig = go.Figure()
    for lo, hi, fill, name in (
        ("q05", "q95", C["band90"], "90% range"),
        ("q25", "q75", C["band50"], "50% range"),
    ):
        fig.add_trace(
            go.Scatter(
                x=x, y=mw(hi), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mw(lo),
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=fill,
                name=name,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mw("q50"),
            mode="lines",
            name="Median forecast",
            line=dict(color=C["s1"], width=2),
            hovertemplate="%{y:.1f} " + title_units,
        )
    )
    if fc["y"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=x,
                y=fc["y"] * fc["ac_mw"],
                mode="lines",
                name="Observed",
                line=dict(color=C["s2"], width=2),
                hovertemplate="%{y:.1f} " + title_units,
            )
        )
    _layout(fig, 340, hovermode="x unified")
    fig.update_yaxes(title_text=f"Power ({title_units})", rangemode="tozero")
    fig.update_xaxes(title_text="Local time (UTC+1)", tickformat="%H:%M")
    return fig


def fig_map(plants: pd.DataFrame) -> go.Figure:
    size = 8 + 22 * np.sqrt(plants["ac_mw"] / plants["ac_mw"].max())
    fig = go.Figure(
        go.Scattergeo(
            lon=plants["longitude"],
            lat=plants["latitude"],
            mode="markers",
            marker=dict(size=size, color=C["s1"], line=dict(color=C["surface"], width=2)),
            text=plants["name"] + " (" + plants["wilaya"] + ")",
            customdata=np.column_stack([plants["ac_mw"].round(1)]),
            hovertemplate="%{text}<br>%{customdata[0]} MW AC<extra></extra>",
        )
    )
    _layout(fig, 420)
    fig.update_geos(
        projection_type="natural earth",
        lataxis_range=[18, 38],
        lonaxis_range=[-9, 13],
        showcountries=True,
        countrycolor=C["axis"],
        showland=True,
        landcolor=C["land"],
        showocean=True,
        oceancolor=C["surface"],
        showcoastlines=True,
        coastlinecolor=C["axis"],
        bgcolor=C["surface"],
        showframe=False,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig


def fig_hbar(labels: list[str], values: list[float], fmt: str, axis_title: str) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=C["s1"], cornerradius=4),
            texttemplate=fmt,
            textposition="outside",
            textfont=dict(color=C["ink2"]),
            hovertemplate="%{y}: " + fmt.replace("%{x", "%{x") + "<extra></extra>",
            cliponaxis=False,
        )
    )
    _layout(fig, 60 + 44 * len(labels), bargap=0.45, showlegend=False)
    lo, hi = min(0.0, min(values)), max(0.0, max(values))
    span = (hi - lo) or 1.0
    # Leave room beyond each bar end so value labels never sit on the axis labels.
    fig.update_xaxes(
        title_text=axis_title, range=[lo - (0.25 * span if lo < 0 else 0), hi + 0.18 * span]
    )
    fig.update_yaxes(autorange="reversed", showgrid=False, automargin=True)
    fig.update_layout(margin=dict(l=8, r=48, t=8, b=48))
    return fig


def fig_calibration(metrics: pd.DataFrame, level: str, target: float = 0.8) -> go.Figure:
    m = metrics[metrics["level"] == level]
    base = sorted(x for x in m["model"].unique() if not x.endswith("+cqr"))
    raw = [float(m.loc[m["model"] == b, "coverage_80"].iloc[0]) for b in base]
    cal = [
        float(m.loc[m["model"] == b + "+cqr", "coverage_80"].iloc[0])
        if (b + "+cqr") in set(m["model"])
        else None
        for b in base
    ]
    names = [MODEL_NAMES.get(b, b) for b in base]
    fig = go.Figure()
    for b, r, c in zip(names, raw, cal, strict=True):
        if c is not None:
            fig.add_trace(
                go.Scatter(
                    x=[r, c],
                    y=[b, b],
                    mode="lines",
                    line=dict(color=C["axis"], width=2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    fig.add_trace(
        go.Scatter(
            x=raw,
            y=names,
            mode="markers",
            name="Raw model",
            marker=dict(size=12, color=C["muted"], line=dict(color=C["surface"], width=2)),
            hovertemplate="%{y}, raw: %{x:.0%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cal,
            y=names,
            mode="markers",
            name="Calibrated",
            marker=dict(size=12, color=C["s1"], line=dict(color=C["surface"], width=2)),
            hovertemplate="%{y}, calibrated: %{x:.0%}<extra></extra>",
        )
    )
    fig.add_vline(x=target, line=dict(color=C["ink2"], width=1))
    fig.add_annotation(
        x=target,
        y=1.0,
        yref="paper",
        text=f"target {target:.0%}",
        showarrow=False,
        yshift=10,
        font=dict(color=C["ink2"], size=12),
    )
    _layout(fig, 90 + 48 * len(base))
    fig.update_xaxes(title_text="Hours inside the 80% range", tickformat=".0%", range=[0.4, 1.0])
    fig.update_yaxes(showgrid=False, automargin=True)
    fig.update_layout(margin=dict(l=8, r=24, t=36, b=48))
    return fig


def _table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    rows = []
    for _, r in frame.iterrows():
        cells = []
        for c in frame.columns:
            v = r[c]
            is_number = isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(
                v, bool
            )
            if pd.isna(v):
                text = ""
            elif c in formats and is_number:
                text = formats[c].format(v)
            else:
                text = str(v)
            cls = ' class="num"' if is_number else ""
            cells.append(f"<td{cls}>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def build_site(settings: Settings) -> Path:
    fc_live = read_serving(settings, "forecasts_live")
    fc = read_serving(settings, "forecasts")
    metrics = read_serving(settings, "metrics")
    regimes = read_serving(settings, "metrics_by_regime")
    plants = read_serving(settings, "plants")
    dispatch = read_serving(settings, "dispatch_summary")
    sensitivity = read_serving(settings, "dispatch_sensitivity")
    recon = read_serving(settings, "reconciliation")
    synthetic = (serving_dir(settings) / "SYNTHETIC").exists()

    figures: dict[str, go.Figure] = {}
    sections: list[str] = []
    best = _best_model(metrics)

    # 1. The headline: tomorrow (live) or the most recent backtest day.
    source = fc_live if not fc_live.empty else fc
    if not source.empty and best:
        day = source[(source["level"] == "country") & (source["model"] == best)]
        day = day[day["issue_time"] == day["issue_time"].max()]
        if not day.empty:
            target = (day["valid_time"].min() + LOCAL).strftime("%A %d %B %Y")
            energy = float((day["q50"] * day["ac_mw"]).sum())
            label = "Tomorrow" if not fc_live.empty else "Most recent backtest day"
            figures["fan"] = fig_fan(day)
            hourly = day.sort_values("valid_time").assign(
                hour=lambda d: (d["valid_time"] + LOCAL).dt.strftime("%H:%M"),
                p10=lambda d: d["q10"] * d["ac_mw"],
                median=lambda d: d["q50"] * d["ac_mw"],
                p90=lambda d: d["q90"] * d["ac_mw"],
            )[["hour", "p10", "median", "p90"]]
            hourly = hourly[hourly["p90"] > 0]
            sections.append(f"""
<section>
  <h2>{label}: national solar output, {html.escape(target)}</h2>
  <p class="lede">Median expected energy <strong class="hero">{energy:,.0f} MWh</strong>
  from {len(plants)} plants, with the ranges that the calibrated <code>{html.escape(best)}</code> model gives.</p>
  <div class="chart" id="fan"></div>
  <details><summary>Hourly values (MW)</summary>{_table(hourly, {c: "{:.1f}" for c in ("p10", "median", "p90")})}</details>
</section>""")

    if not plants.empty:
        figures["map"] = fig_map(plants)
        sections.append(f"""
<section>
  <h2>The plants</h2>
  <p>Marker size shows AC capacity. Locations and capacities come from public sources; see <code>config/plants.csv</code>.</p>
  <div class="chart" id="map"></div>
  <details><summary>Plant table</summary>{_table(plants[["name", "wilaya", "ac_mw", "coord_quality", "status"]].rename(columns={"ac_mw": "AC MW"}), {"AC MW": "{:.1f}"})}</details>
</section>""")

    if not metrics.empty:
        country = metrics[(metrics["level"] == "country") & metrics["model"].str.endswith("+cqr")]
        country = country.sort_values("skill_vs_persistence", ascending=False)
        figures["skill"] = fig_hbar(
            [MODEL_NAMES.get(m.removesuffix("+cqr"), m) for m in country["model"]],
            country["skill_vs_persistence"].tolist(),
            "%{x:.0%}",
            "Skill vs persistence",
        )
        cols = [
            "model",
            "level",
            "quantile_score",
            "mae",
            "coverage_80",
            "width_80",
            "coverage_90",
            "skill_vs_persistence",
        ]
        table = (
            metrics[cols]
            .sort_values(["level", "quantile_score"])
            .assign(model=lambda d: d["model"].map(pretty))
        )
        pct = {c: "{:.0%}" for c in ("coverage_80", "coverage_90", "skill_vs_persistence")}
        sections.append(f"""
<section>
  <h2>How good are the forecasts?</h2>
  <p>Backtest over every issue day after the calibration warm-up, daylight hours only. Skill is how much lower
  each calibrated model's quantile score is than persistence's, at the national level (higher is better).</p>
  <div class="chart" id="skill"></div>
  <details><summary>All scores</summary>{_table(table, {"quantile_score": "{:.4f}", "mae": "{:.4f}", "width_80": "{:.3f}", **pct})}</details>
</section>""")
        figures["calibration"] = fig_calibration(metrics, "plant")
        reg = (
            regimes[(regimes["level"] == "plant") & regimes["model"].str.endswith("+cqr")]
            if not regimes.empty
            else regimes
        )
        sections.append(f"""
<section>
  <h2>Are the uncertainty ranges honest?</h2>
  <p>An 80% range should contain the observation 80% of the time. Grey dots are the raw models, blue dots the
  same models after conformal calibration (plant level).</p>
  <div class="chart" id="calibration"></div>
  <details><summary>Coverage by type of day</summary>{_table(reg[["model", "regime", "n_hours", "coverage_80", "coverage_90", "quantile_score"]].assign(model=lambda d: d["model"].map(pretty)), {"coverage_80": "{:.0%}", "coverage_90": "{:.0%}", "quantile_score": "{:.4f}"}) if not reg.empty else ""}</details>
</section>""")

    if not dispatch.empty:
        d = dispatch[dispatch["policy"] != "perfect_foresight"]
        figures["dispatch"] = fig_hbar(
            d["policy"].map(POLICY_NAMES).tolist(),
            d["regret_per_mw_day"].tolist(),
            "%{x:.2f}",
            "Regret per MW per day",
        )
        sens = ""
        if not sensitivity.empty:
            sens = f"<details><summary>Sensitivity to battery size and penalty</summary>{_table(sensitivity, {'regret_per_mw_day': '{:.3f}'})}</details>"
        sections.append(f"""
<section>
  <h2>What are the forecasts worth?</h2>
  <p>Each plant commits tomorrow's hourly delivery; every missing MWh costs {settings.battery.shortfall_penalty:g}&times;
  the energy price. Regret is the profit lost compared with knowing tomorrow's output in advance (lower is
  better). "Scenarios" plans over many possible days drawn from the forecast ranges; "Median" plans as if the
  median were certain.</p>
  <div class="chart" id="dispatch"></div>
  <details><summary>Dispatch results</summary>{_table(dispatch.assign(policy=dispatch["policy"].map(POLICY_NAMES)), {c: "{:.3f}" for c in dispatch.columns if dispatch[c].dtype.kind == "f"})}</details>
  {sens}
</section>""")

    if not recon.empty:
        sections.append(f"""
<section>
  <h2>Do plant, wilaya and national forecasts add up?</h2>
  <p>Mean absolute error of the median (MW, daylight hours) before and after reconciliation.</p>
  {_table(recon.drop(columns=["model"], errors="ignore").rename(columns=RECON_NAMES), {v: "{:.3f}" for k, v in RECON_NAMES.items() if k != "level"})}
</section>""")

    banner = ""
    if synthetic:
        banner = (
            '<p class="banner" role="note"><span aria-hidden="true">&#9888;</span> <strong>Synthetic data.</strong> '
            "This page was built by the offline demo. None of these numbers describe real plants.</p>"
        )
    payload = {k: json.loads(pio.to_json(f)) for k, f in figures.items()}
    page = PAGE.format(
        banner=banner,
        sections="\n".join(sections)
        or "<p>No results yet. Run <code>sahelcast pipeline</code>.</p>",
        figures=json.dumps(payload),
        dark=json.dumps(DARK),
        pattern=json.dumps("|".join(re.escape(k) for k in DARK)),
        generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        plotly_version=get_plotlyjs_version(),
    )
    out = Path(settings.paths.site)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page, encoding="utf-8")
    return out / "index.html"


def _best_model(metrics: pd.DataFrame) -> str | None:
    if metrics.empty:
        return None
    m = metrics[(metrics["level"] == "country") & metrics["model"].str.endswith("+cqr")]
    return None if m.empty else str(m.sort_values("quantile_score")["model"].iloc[0])


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SahelCast</title>
<script src="https://cdn.plot.ly/plotly-{plotly_version}.min.js" charset="utf-8"></script>
<style>
:root {{
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --ring: rgba(11,11,11,0.10); --warn: #fab219;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --ring: rgba(255,255,255,0.10);
  }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--page); color: var(--ink);
  font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 980px; margin: 0 auto; padding: 32px 16px 64px; }}
h1 {{ font-size: 2rem; margin: 0 0 4px; }}
h2 {{ font-size: 1.25rem; margin: 0 0 8px; }}
.sub {{ color: var(--ink2); margin: 0 0 24px; }}
section {{ background: var(--surface); border: 1px solid var(--ring); border-radius: 12px;
  padding: 20px; margin: 0 0 20px; }}
section p {{ color: var(--ink2); margin: 0 0 12px; }}
.lede {{ color: var(--ink); }}
.hero {{ font-size: 1.6rem; font-weight: 600; }}
.chart {{ width: 100%; min-height: 120px; }}
details {{ margin-top: 12px; }}
summary {{ cursor: pointer; color: var(--ink2); }}
.table-wrap {{ overflow-x: auto; margin-top: 8px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.875rem; }}
th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid); white-space: nowrap; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
th {{ color: var(--ink2); font-weight: 600; }}
code {{ font-size: 0.9em; }}
.banner {{ border-left: 4px solid var(--warn); background: var(--surface); padding: 12px 16px;
  border-radius: 8px; margin: 0 0 20px; }}
footer {{ color: var(--muted); font-size: 0.875rem; }}
footer a {{ color: var(--ink2); }}
</style>
</head>
<body>
<main>
  <h1>SahelCast</h1>
  <p class="sub">Day-ahead solar power forecasts with calibrated uncertainty for Algerian PV plants,
  and what they are worth for battery scheduling.</p>
  {banner}
  {sections}
  <footer>
    <p>Weather, satellite irradiance and reanalysis data: <a href="https://open-meteo.com/">Open-Meteo.com</a>
    (CC BY 4.0), which redistributes ECMWF, EUMETSAT and DWD products; satellite irradiance is the
    SARAH-3 climate data record from EUMETSAT CM SAF. Plant list: Global Energy Monitor wiki
    and press reports. Plant output is simulated from satellite irradiance with pvlib; it is not metered data.
    Generated {generated}.</p>
  </footer>
</main>
<script>
const FIGURES = {figures};
const DARK = {dark};
const media = window.matchMedia("(prefers-color-scheme: dark)");
const PATTERN = new RegExp({pattern}, "gi");
function themed(fig) {{
  // One pass swaps every light colour for its dark twin (no chained replacements).
  if (!media.matches) return fig;
  return JSON.parse(JSON.stringify(fig).replace(PATTERN, m => DARK[m.toLowerCase()] || m));
}}
function draw() {{
  for (const [id, fig] of Object.entries(FIGURES)) {{
    const el = document.getElementById(id);
    if (!el) continue;
    const f = themed(fig);
    Plotly.react(el, f.data, f.layout, {{responsive: true, displayModeBar: false}});
  }}
}}
draw();
media.addEventListener("change", draw);
</script>
</body>
</html>
"""
