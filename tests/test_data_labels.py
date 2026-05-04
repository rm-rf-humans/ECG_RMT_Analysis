from __future__ import annotations

import pandas as pd

from ecg_rmt.data import attach_binary_target, parse_scp_codes, supervised_records


def test_parse_scp_codes_string() -> None:
    assert parse_scp_codes("{'NORM': 100.0, 'MI': 50.0}") == {
        "NORM": 100.0,
        "MI": 50.0,
    }


def test_attach_binary_target() -> None:
    metadata = pd.DataFrame(
        {
            "ecg_id": [1, 2, 3],
            "scp_codes": [{"NORM": 100.0}, {"MI": 100.0}, {"UNKNOWN": 50.0}],
        }
    )
    statements = pd.DataFrame(
        {
            "diagnostic": [1, 1],
            "diagnostic_class": ["NORM", "MI"],
        },
        index=["NORM", "MI"],
    )
    labeled = attach_binary_target(metadata, statements)
    supervised = supervised_records(labeled)
    assert supervised["target"].tolist() == [0, 1]
    assert supervised["target_name"].tolist() == ["normal", "diagnostic_abnormality"]


def test_multilabel_with_norm_and_abnormal_is_positive() -> None:
    metadata = pd.DataFrame({"ecg_id": [1], "scp_codes": [{"NORM": 50.0, "CD": 80.0}]})
    statements = pd.DataFrame(
        {"diagnostic": [1, 1], "diagnostic_class": ["NORM", "CD"]},
        index=["NORM", "CD"],
    )
    labeled = attach_binary_target(metadata, statements)
    assert int(labeled.loc[0, "target"]) == 1
