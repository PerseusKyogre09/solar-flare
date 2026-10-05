import numpy as np
import pandas as pd

from src.xgb_baseline.data import parse_filename, split_overlap, target_from_flare
from src.xgb_baseline.metrics import binary_metrics


def test_target_conversion():
    labels = pd.Series(["0", "C1.0", "M2.0", "X1.0"])
    assert target_from_flare(labels, "ge_c").tolist() == [0, 1, 1, 1]
    assert target_from_flare(labels, "ge_m").tolist() == [0, 0, 1, 1]


def test_filename_parsing():
    ar, ts = parse_filename(pd.Series(["1069/1069_hmi.M_720s.20100505_001200_TAI.1.magnetogram_224.png"]))
    assert ar.iloc[0] == "1069" and str(ts.iloc[0]) == "2010-05-05 00:12:00"


def test_split_overlap():
    a = pd.DataFrame({"active_region": ["1", "2"]}); b = pd.DataFrame({"active_region": ["2", "3"]})
    assert split_overlap({"Train": a, "Validation": b})["Train_Validation"] == ["2"]


def test_metrics_manual():
    result = binary_metrics([1, 1, 1, 0, 0, 0, 0, 0], [0.9, 0.8, 0.7, 0.6, 0.1, 0.1, 0.1, 0.1], 0.5)
    assert (result["tp"], result["tn"], result["fp"], result["fn"]) == (3, 4, 1, 0)
    assert result["pod_recall_tpr"] == 1.0 and result["far"] == 0.25
    assert result["tss"] == 0.75  # TPR - FPR, not TPR - TNR
    assert np.isfinite(result["hss"]) and np.isfinite(result["tss"])
