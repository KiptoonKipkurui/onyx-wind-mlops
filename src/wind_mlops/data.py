from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


SCADA_COLUMNS = [
    "Wind speed (m/s)",
    "Wind speed, Standard deviation (m/s)",
    "Density adjusted wind speed (m/s)",
    "Wind direction (Â°)",
    "Nacelle position (Â°)",
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

LEAKAGE_COLUMNS = [
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
]

EVENT_PRIORITY = [
    ("mechanical_issue", "active_mechanical"),
    ("grid_or_external", "active_grid_or_external"),
    ("forced_outage", "active_forced_outage"),
    ("scheduled_maintenance", "active_scheduled_maintenance"),
    ("stop_other", "active_stop"),
    ("warning", "active_warning"),
]


def turbine_number(path: Path) -> str:
    match = re.search(r"Penmanshiel_(\d{2})_", path.name)
    if not match:
        raise ValueError(f"Could not infer turbine number from {path.name}")
    return match.group(1)


def find_data_dir(raw_dir: Path) -> Path:
    if any(raw_dir.glob("Turbine_Data_*.csv")):
        return raw_dir
    candidates = [path for path in raw_dir.rglob("*") if path.is_dir() and any(path.glob("Turbine_Data_*.csv"))]
    if not candidates:
        raise FileNotFoundError(
            f"No Penmanshiel turbine CSV files found under {raw_dir}. "
            "Expected files named Turbine_Data_*.csv and Status_*.csv."
        )
    return sorted(candidates)[0]


def read_scada_file(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    header = pd.read_csv(path, skiprows=9, nrows=0)
    available = [col for col in SCADA_COLUMNS if col in header.columns]
    usecols = ["# Date and time", *available]
    df = pd.read_csv(path, skiprows=9, usecols=usecols, nrows=max_rows, low_memory=False)
    df = df.rename(columns={"# Date and time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["turbine"] = f"T{turbine_number(path)}"
    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["timestamp"])


def read_status_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=9, low_memory=False).drop_duplicates()
    df["start"] = pd.to_datetime(df["Timestamp start"], errors="coerce")
    df["end"] = pd.to_datetime(df["Timestamp end"], errors="coerce")
    df["duration_hours"] = (df["end"] - df["start"]).dt.total_seconds() / 3600
    df["turbine"] = f"T{turbine_number(path)}"
    return df.dropna(subset=["start", "end"])


def interval_mask(times: pd.Series, intervals: pd.DataFrame) -> np.ndarray:
    if intervals.empty:
        return np.zeros(len(times), dtype=bool)

    ordered = times.sort_values()
    starts = np.searchsorted(ordered.values, intervals["start"].values, side="left")
    ends = np.searchsorted(ordered.values, intervals["end"].values, side="left")
    deltas = np.zeros(len(times) + 1, dtype=np.int32)
    for start, end in zip(starts, ends, strict=False):
        if end > start:
            deltas[start] += 1
            deltas[end] -= 1

    active_sorted = np.cumsum(deltas[:-1]) > 0
    return pd.Series(active_sorted, index=ordered.index).reindex(times.index).to_numpy(dtype=bool)


def attach_status_flags(scada: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    service = status["Service contract category"].astype(str)
    iec = status["IEC category"].astype(str)
    status_name = status["Status"].astype(str).str.casefold()
    selectors = {
        "active_stop": status_name.eq("stop"),
        "active_warning": status_name.eq("warning"),
        "active_forced_outage": iec.str.contains("forced outage", case=False, na=False),
        "active_scheduled_maintenance": iec.str.contains("scheduled maintenance", case=False, na=False),
        "active_grid_or_external": service.str.contains("grid|external", case=False, na=False),
        "active_mechanical": service.str.contains("mechanical", case=False, na=False),
    }

    out = scada.copy()
    for flag, selector in selectors.items():
        out[flag] = interval_mask(out["timestamp"], status.loc[selector])
    return out


def load_penmanshiel_dataset(data_dir: Path, max_rows_per_turbine: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = data_dir if any(data_dir.glob("Turbine_Data_*.csv")) else find_data_dir(data_dir)
    scada_frames: list[pd.DataFrame] = []
    status_frames: list[pd.DataFrame] = []
    for scada_path in sorted(root.glob("Turbine_Data_*.csv")):
        number = turbine_number(scada_path)
        status_matches = sorted(root.glob(f"Status_Penmanshiel_{number}_*.csv"))
        if not status_matches:
            raise FileNotFoundError(f"No matching status file found for {scada_path.name}")
        status = read_status_file(status_matches[0])
        scada = read_scada_file(scada_path, max_rows=max_rows_per_turbine)
        scada_frames.append(attach_status_flags(scada, status))
        status_frames.append(status)

    if not scada_frames:
        raise FileNotFoundError(f"No turbine files found in {root}")
    return pd.concat(scada_frames, ignore_index=True), pd.concat(status_frames, ignore_index=True)


def build_multiclass_training_frame(
    combined: pd.DataFrame,
    future_steps: int = 6,
    min_class_rows: int = 30,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], dict[str, int]]:
    model_data = combined.sort_values(["turbine", "timestamp"]).copy()
    future_event_masks: dict[str, np.ndarray] = {}
    for class_name, flag in EVENT_PRIORITY:
        shifted_flags = []
        for step in range(1, future_steps + 1):
            shifted = model_data.groupby("turbine")[flag].shift(-step)
            shifted = shifted.where(shifted.notna(), False).astype(bool)
            shifted_flags.append(shifted.to_numpy())
        future_event_masks[class_name] = np.column_stack(shifted_flags).any(axis=1)

    next_event_type = np.full(len(model_data), "no_event", dtype=object)
    for class_name, _ in reversed(EVENT_PRIORITY):
        next_event_type[future_event_masks[class_name]] = class_name
    model_data["next_event_type_60m"] = next_event_type

    current_major_event = (
        model_data["active_stop"]
        | model_data["active_forced_outage"]
        | model_data["active_scheduled_maintenance"]
        | model_data["active_grid_or_external"]
        | model_data["active_mechanical"]
    )
    model_data = model_data[~current_major_event].copy()

    feature_names = [
        col for col in SCADA_COLUMNS if col in model_data.columns and col not in LEAKAGE_COLUMNS
    ]
    feature_names = [
        col
        for col in feature_names
        if model_data[col].notna().mean() >= 0.35 and model_data[col].nunique(dropna=True) > 2
    ]

    class_counts = model_data["next_event_type_60m"].value_counts()
    kept_classes = class_counts[class_counts >= min_class_rows].index.tolist()
    model_data = model_data[model_data["next_event_type_60m"].isin(kept_classes)].copy()

    X = model_data[feature_names].replace([np.inf, -np.inf], np.nan).astype("float32")
    y = model_data["next_event_type_60m"].astype(str)
    class_names = sorted(y.unique().tolist())
    class_to_id = {name: index for index, name in enumerate(class_names)}
    return X, y, feature_names, class_names, class_to_id
