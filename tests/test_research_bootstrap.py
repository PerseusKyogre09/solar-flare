import numpy as np

from src.research_metrics import (
    binary_curve_data,
    cluster_bootstrap_metric_samples,
    expected_calibration_error,
    map_feature_groups,
    make_cluster_bootstrap_counts,
    multiclass_cluster_bootstrap_metric_samples,
    percentile_intervals,
    sklearn_weighted_rank_metrics,
)
from src.xgb_baseline.metrics import binary_metrics


def test_weighted_rank_metrics_match_sklearn_with_score_ties():
    y = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    probability = np.array([0.9, 0.9, 0.6, 0.6, 0.6, 0.2, 0.2, 0.1])
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    _, row_codes, draws = make_cluster_bootstrap_counts(groups, n_resamples=8, seed=42)

    samples = cluster_bootstrap_metric_samples(y, probability, row_codes, draws, 0.5)

    for index, draw in enumerate(draws):
        weights = draw[row_codes]
        if np.unique(y[weights > 0]).size == 2:
            expected_roc, expected_pr = sklearn_weighted_rank_metrics(y, probability, weights)
            assert np.isclose(samples["roc_auc"][index], expected_roc)
            assert np.isclose(samples["pr_auc"][index], expected_pr)
        else:
            assert np.isnan(samples["roc_auc"][index])
            assert np.isnan(samples["pr_auc"][index])


def test_threshold_metrics_match_weighted_confusion_counts():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    probability = np.array([0.9, 0.6, 0.8, 0.2, 0.7, 0.4, 0.3, 0.1])
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    _, row_codes, draws = make_cluster_bootstrap_counts(groups, n_resamples=1, seed=11)
    weights = draws[0][row_codes]

    samples = cluster_bootstrap_metric_samples(y, probability, row_codes, draws, 0.5)
    expected = binary_metrics(np.repeat(y, weights), np.repeat(probability, weights), 0.5)

    for key in ("tss", "hss", "csi", "mcc", "f1", "precision", "pod_recall_tpr", "specificity", "far", "brier_score"):
        assert np.isclose(samples[key][0], expected[key])


def test_cluster_draws_and_percentile_intervals_are_reproducible():
    groups = np.array(["ar-1", "ar-1", "ar-2", "ar-3"])
    _, first_codes, first_draws = make_cluster_bootstrap_counts(groups, n_resamples=20, seed=42)
    _, second_codes, second_draws = make_cluster_bootstrap_counts(groups, n_resamples=20, seed=42)
    assert np.array_equal(first_codes, second_codes)
    assert np.array_equal(first_draws, second_draws)
    interval = percentile_intervals({"metric": np.arange(100, dtype=float)})["metric"]
    assert len(interval) == 2 and interval[0] < interval[1]


def test_equal_width_expected_calibration_error():
    assert np.isclose(expected_calibration_error([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]), 0.15)


def test_roc_and_pr_coordinates_come_from_actual_probabilities():
    y = np.array([0, 1, 0, 1])
    probability = np.array([0.1, 0.9, 0.4, 0.6])
    false_positive_rate, true_positive_rate = binary_curve_data(y, probability, "roc")
    recall, precision = binary_curve_data(y, probability, "pr")
    assert np.array_equal(false_positive_rate[[0, -1]], [0.0, 1.0])
    assert np.array_equal(true_positive_rate[[0, -1]], [0.0, 1.0])
    assert recall[0] == 1.0 and recall[-1] == 0.0
    assert precision[-1] == 1.0


def test_feature_group_mapping_refuses_generic_columns():
    assert map_feature_groups([f"feature_{index:02d}" for index in range(29)]) is None
    mapped = map_feature_groups(["mean_gradient", "total_flux", "wavelet_energy", "pil_length"])
    assert mapped == {
        "mean_gradient": "magnetic_gradient",
        "total_flux": "flux",
        "wavelet_energy": "wavelet",
        "pil_length": "neutral_line_pil",
    }


def test_multiclass_cluster_bootstrap_metrics_are_perfect_for_perfect_predictions():
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    groups = np.array(["a"] * 3 + ["b"] * 3 + ["c"] * 3)
    _, codes, draws = make_cluster_bootstrap_counts(groups, n_resamples=20, seed=42)
    samples = multiclass_cluster_bootstrap_metric_samples(y, y, codes, draws)
    for metric, values in samples.items():
        expected = 0.0 if metric == "macro_far" else 1.0
        assert np.allclose(values, expected)