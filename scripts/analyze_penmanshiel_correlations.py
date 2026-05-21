from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


SCADA_COLUMNS = [
    "Wind speed (m/s)",
    "Wind speed, Standard deviation (m/s)",
    "Density adjusted wind speed (m/s)",
    "Power (kW)",
    "Potential power default PC (kW)",
    "Available Capacity for Production (kW)",
    "Energy Export (kWh)",
    "Lost Production to Downtime (kWh)",
    "Lost Production to Performance (kWh)",
    "Lost Production Total (kWh)",
    "Capacity factor",
    "Data Availability",
    "Time-based System Avail.",
    "Production-based System Avail.",
    "Performance Index",
    "Rotor speed (RPM)",
    "Generator RPM (RPM)",
    "Front bearing temperature (Â°C)",
    "Rear bearing temperature (Â°C)",
    "Stator temperature 1 (Â°C)",
    "Nacelle ambient temperature (Â°C)",
    "Nacelle temperature (Â°C)",
    "Transformer temperature (Â°C)",
    "Gear oil temperature (Â°C)",
    "Generator bearing rear temperature (Â°C)",
    "Generator bearing front temperature (Â°C)",
    "Grid voltage (V)",
    "Grid current (A)",
    "Reactive power (kvar)",
    "Blade angle (pitch position) A (Â°)",
    "Blade angle (pitch position) B (Â°)",
    "Blade angle (pitch position) C (Â°)",
    "Gear oil inlet pressure (bar)",
    "Gear oil pump pressure (bar)",
    "Grid frequency (Hz)",
    "Drive train acceleration (mm/ss)",
    "Tower Acceleration X (mm/ss)",
    "Tower Acceleration y (mm/ss)",
]


def turbine_number(path: Path) -> str:
    match = re.search(r"Penmanshiel_(\d{2})_", path.name)
    if not match:
        raise ValueError(f"Could not infer turbine number from {path.name}")
    return match.group(1)


def read_scada(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, skiprows=9, nrows=0)
    available = [col for col in SCADA_COLUMNS if col in header.columns]
    usecols = ["# Date and time", *available]
    df = pd.read_csv(path, skiprows=9, usecols=usecols, low_memory=False)
    df = df.rename(columns={"# Date and time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["turbine"] = f"T{turbine_number(path)}"
    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["timestamp"])


def read_status(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=9, low_memory=False)
    df = df.drop_duplicates()
    df["start"] = pd.to_datetime(df["Timestamp start"], errors="coerce")
    df["end"] = pd.to_datetime(df["Timestamp end"], errors="coerce")
    df["duration_hours"] = (df["end"] - df["start"]).dt.total_seconds() / 3600
    df["turbine"] = f"T{turbine_number(path)}"
    return df.dropna(subset=["start", "end"])


def interval_mask(times: pd.Series, intervals: pd.DataFrame) -> np.ndarray:
    if intervals.empty:
        return np.zeros(len(times), dtype=bool)
    ordered_times = times.sort_values()
    starts = np.searchsorted(ordered_times.values, intervals["start"].values, side="left")
    ends = np.searchsorted(ordered_times.values, intervals["end"].values, side="left")
    deltas = np.zeros(len(times) + 1, dtype=np.int32)
    for start, end in zip(starts, ends, strict=False):
        if end > start:
            deltas[start] += 1
            deltas[end] -= 1
    active_sorted = np.cumsum(deltas[:-1]) > 0
    active = pd.Series(active_sorted, index=ordered_times.index).reindex(times.index)
    return active.to_numpy(dtype=bool)


def attach_status_flags(scada: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    flags = {
        "active_stop": status["Status"].astype(str).str.casefold().eq("stop"),
        "active_warning": status["Status"].astype(str).str.casefold().eq("warning"),
        "active_forced_outage": status["IEC category"].astype(str).str.contains(
            "forced outage", case=False, na=False
        ),
        "active_scheduled_maintenance": status["IEC category"].astype(str).str.contains(
            "scheduled maintenance", case=False, na=False
        ),
        "active_grid_or_external": status["Service contract category"].astype(str).str.contains(
            "grid|external", case=False, na=False
        ),
        "active_mechanical": status["Service contract category"].astype(str).str.contains(
            "mechanical", case=False, na=False
        ),
    }
    for name, selector in flags.items():
        scada[name] = interval_mask(scada["timestamp"], status.loc[selector])
    return scada


def top_correlations(df: pd.DataFrame, target: str, top_n: int = 12) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in SCADA_COLUMNS
        if col in df.columns and df[col].notna().mean() >= 0.35 and df[col].nunique(dropna=True) > 2
    ]
    rows = []
    y = df[target].astype(float)
    for col in numeric_cols:
        pair = pd.concat([df[col], y], axis=1).dropna()
        if len(pair) < 200 or pair[target].nunique() < 2:
            continue
        corr = pair[col].corr(pair[target])
        if pd.isna(corr):
            continue
        rows.append(
            {
                "target": target,
                "signal": col,
                "correlation": corr,
                "abs_correlation": abs(corr),
                "mean_when_inactive": pair.loc[pair[target] == 0, col].mean(),
                "mean_when_active": pair.loc[pair[target] == 1, col].mean(),
                "rows": len(pair),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).head(top_n)


def summarize_status(status: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_turbine = (
        status.groupby(["turbine", "Status"], dropna=False)["duration_hours"]
        .sum()
        .reset_index()
        .sort_values(["turbine", "duration_hours"], ascending=[True, False])
    )
    by_code = (
        status.groupby(["Code", "Message", "Status"], dropna=False)["duration_hours"]
        .sum()
        .reset_index()
        .sort_values("duration_hours", ascending=False)
        .head(15)
    )
    return by_turbine, by_code


def write_markdown(
    output: Path,
    combined: pd.DataFrame,
    corr: pd.DataFrame,
    by_turbine: pd.DataFrame,
    by_code: pd.DataFrame,
) -> None:
    def md_table(frame: pd.DataFrame, floatfmt: str = ".2f") -> str:
        if frame.empty:
            return "_No rows._"
        display = frame.copy()
        for col in display.select_dtypes(include=[np.number]).columns:
            display[col] = display[col].map(lambda value: format(value, floatfmt))
        headers = [str(col) for col in display.columns]
        rows = [[str(value) for value in row] for row in display.to_numpy()]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    lines = [
        "# Penmanshiel Correlation Summary",
        "",
        f"Rows analysed: {len(combined):,}",
        f"Turbines analysed: {', '.join(sorted(combined['turbine'].unique()))}",
        "",
        "## Event Prevalence",
        "",
    ]
    flag_cols = [col for col in combined.columns if col.startswith("active_")]
    prevalence = pd.DataFrame(
        {
            "flag": flag_cols,
            "active_rows": [int(combined[col].sum()) for col in flag_cols],
            "active_pct": [float(combined[col].mean() * 100) for col in flag_cols],
        }
    )
    lines.append(md_table(prevalence, ".2f"))
    lines += ["", "## Top SCADA Correlations With Status Flags", ""]
    for target in corr["target"].drop_duplicates():
        lines += [f"### {target}", ""]
        subset = corr.loc[corr["target"] == target].drop(columns=["abs_correlation"])
        lines.append(md_table(subset, ".3f"))
        lines.append("")
    lines += ["## Status Duration By Turbine And Type", ""]
    lines.append(md_table(by_turbine, ".2f"))
    lines += ["", "## Longest Status Codes By Duration", ""]
    lines.append(md_table(by_code, ".2f"))
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scada_frames = []
    status_frames = []
    for scada_path in sorted(args.data_dir.glob("Turbine_Data_*.csv")):
        t = turbine_number(scada_path)
        status_path = next(args.data_dir.glob(f"Status_Penmanshiel_{t}_*.csv"))
        scada = read_scada(scada_path)
        status = read_status(status_path)
        scada_frames.append(attach_status_flags(scada, status))
        status_frames.append(status)

    combined = pd.concat(scada_frames, ignore_index=True)
    statuses = pd.concat(status_frames, ignore_index=True)
    by_turbine, by_code = summarize_status(statuses)

    targets = [col for col in combined.columns if col.startswith("active_")]
    corr = pd.concat([top_correlations(combined, target) for target in targets], ignore_index=True)

    combined_profile = {
        "rows": int(len(combined)),
        "turbines": sorted(combined["turbine"].unique().tolist()),
        "timestamp_min": str(combined["timestamp"].min()),
        "timestamp_max": str(combined["timestamp"].max()),
        "status_rows_after_dedup": int(len(statuses)),
    }

    corr.to_csv(args.output_dir / "status_signal_correlations.csv", index=False)
    by_turbine.to_csv(args.output_dir / "status_duration_by_turbine.csv", index=False)
    by_code.to_csv(args.output_dir / "top_status_codes_by_duration.csv", index=False)
    (args.output_dir / "profile.json").write_text(json.dumps(combined_profile, indent=2), encoding="utf-8")
    write_markdown(args.output_dir / "correlation_summary.md", combined, corr, by_turbine, by_code)
    print(json.dumps(combined_profile, indent=2))


if __name__ == "__main__":
    main()
