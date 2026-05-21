from __future__ import annotations

from pathlib import Path

import pandas as pd

from wind_mlops.data import (
    build_multiclass_training_frame,
    load_penmanshiel_dataset,
    read_scada_file,
    read_status_file,
)


SCADA_HEADER = [
    "# Date and time",
    "Wind speed (m/s)",
    "Power (kW)",
    "Rotor speed (RPM)",
    "Generator RPM (RPM)",
    "Grid frequency (Hz)",
    "Available Capacity for Production (kW)",
]

STATUS_HEADER = [
    "Timestamp start",
    "Timestamp end",
    "Duration",
    "Status",
    "Code",
    "Message",
    "Comment",
    "Service contract category",
    "IEC category",
]


def write_greenbyte_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    metadata = "\n".join([f"metadata line {index}" for index in range(1, 10)])
    body = "\n".join([",".join(header), *[",".join(map(str, row)) for row in rows]])
    path.write_text(f"{metadata}\n{body}\n", encoding="utf-8")


def test_readers_and_loader_pair_turbine_data_with_status(tmp_path: Path) -> None:
    scada_path = tmp_path / "Turbine_Data_Penmanshiel_01_2016-01-01_-_2017-01-01_1042.csv"
    status_path = tmp_path / "Status_Penmanshiel_01_2016-01-01_-_2017-01-01_1042.csv"
    write_greenbyte_csv(
        scada_path,
        SCADA_HEADER,
        [
            ["2016-01-01 00:00:00", 5.0, 100, 8, 900, 50.0, 2050],
            ["2016-01-01 00:10:00", 6.0, 150, 9, 910, 50.1, 2050],
            ["2016-01-01 00:20:00", 7.0, 200, 10, 920, 49.9, 2050],
        ],
    )
    write_greenbyte_csv(
        status_path,
        STATUS_HEADER,
        [
            [
                "2016-01-01 00:10:00",
                "2016-01-01 00:30:00",
                "00:20:00",
                "Stop",
                101,
                "Forced outage stop",
                "",
                "Mechanical",
                "Forced outage",
            ]
        ],
    )

    scada = read_scada_file(scada_path)
    status = read_status_file(status_path)
    combined, loaded_status = load_penmanshiel_dataset(tmp_path)

    assert scada["turbine"].unique().tolist() == ["T01"]
    assert len(status) == 1
    assert status.loc[0, "duration_hours"] == 1 / 3
    assert len(combined) == 3
    assert len(loaded_status) == 1
    assert combined["active_stop"].tolist() == [False, True, True]
    assert combined["active_forced_outage"].tolist() == [False, True, True]
    assert combined["active_mechanical"].tolist() == [False, True, True]


def test_build_multiclass_training_frame_uses_future_status_and_excludes_leakage() -> None:
    timestamps = pd.date_range("2016-01-01 00:00:00", periods=12, freq="10min")
    combined = pd.DataFrame(
        {
            "timestamp": timestamps,
            "turbine": ["T01"] * len(timestamps),
            "Wind speed (m/s)": range(1, 13),
            "Power (kW)": range(100, 220, 10),
            "Rotor speed (RPM)": range(8, 20),
            "Generator RPM (RPM)": range(900, 1020, 10),
            "Grid frequency (Hz)": [49.8, 49.9, 50.0, 50.1] * 3,
            "Available Capacity for Production (kW)": [2050] * len(timestamps),
            "active_stop": [False] * len(timestamps),
            "active_warning": [False] * len(timestamps),
            "active_forced_outage": [False] * len(timestamps),
            "active_scheduled_maintenance": [False] * len(timestamps),
            "active_grid_or_external": [False] * len(timestamps),
            "active_mechanical": [False] * len(timestamps),
        }
    )
    combined.loc[3, "active_mechanical"] = True
    combined.loc[8, "active_grid_or_external"] = True
    combined.loc[10, "active_warning"] = True

    X, y, feature_names, class_names, class_to_id = build_multiclass_training_frame(
        combined,
        future_steps=2,
        min_class_rows=1,
    )

    assert "Available Capacity for Production (kW)" not in feature_names
    assert "Wind speed (m/s)" in feature_names
    assert X.shape[0] == len(y)
    assert {"mechanical_issue", "grid_or_external", "warning", "no_event"}.issubset(class_names)
    assert class_to_id == {name: index for index, name in enumerate(class_names)}
    assert y.iloc[1] == "mechanical_issue"
    assert y.iloc[6] == "grid_or_external"
    assert "mechanical_issue" not in y.iloc[3:4].tolist()
