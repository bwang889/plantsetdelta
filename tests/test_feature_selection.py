import pandas as pd

from plantsetdelta.feature_selection import CorrelationTopKSelector


def test_selector_uses_only_fit_rows_and_keeps_expected_feature():
    train = pd.DataFrame({
        "signal": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "noise":  [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
        "constant": [0] * 10,
    })
    labels = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    selector = CorrelationTopKSelector(sample_fraction=0.2).fit(train, labels)
    assert "signal" in selector.selected_features_
    assert "constant" not in selector.selected_features_
    validation = pd.DataFrame({"signal": [100, -100], "noise": [0, 1], "constant": [0, 0]})
    transformed = selector.transform(validation)
    assert list(transformed.columns) == selector.selected_features_


def test_selector_supports_nominal_multiclass_labels():
    data = pd.DataFrame({
        "class_signal": [0, 0, 1, 1, 2, 2, 0, 1, 2],
        "noise": [1, 2, 0, 2, 1, 0, 0, 1, 2],
        "constant": [1] * 9,
    })
    labels = pd.Series(["A", "A", "B", "B", "C", "C", "A", "B", "C"])
    selector = CorrelationTopKSelector(sample_fraction=0.3).fit(data, labels)
    assert "class_signal" in selector.selected_features_
    assert "constant" not in selector.selected_features_
