import numpy as np
from src.xgb_baseline.metrics import binary_metrics, select_validation_threshold

def test_binary_skill_scores_and_mcc():
    m = binary_metrics([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], 0.5)
    assert m['tss'] == 1.0 and m['hss'] == 1.0 and m['csi'] == 1.0
    assert m['specificity'] == 1.0
    assert abs(m['mcc'] - 1.0) < 1e-12 if 'mcc' in m else True

def test_threshold_is_selected_from_validation_scores():
    y = np.array([1, 1, 0, 0]); p = np.array([.6, .4, .3, .2])
    assert select_validation_threshold(y, p, [.1, .5, .9]) == .5

def test_degenerate_rates_are_finite():
    m = binary_metrics([0, 0], [0.1, 0.2], .5)
    for key in ('tss','hss','csi','far','specificity','precision','f1','brier_score'):
        assert np.isfinite(m[key])
