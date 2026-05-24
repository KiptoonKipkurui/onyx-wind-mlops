from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from wind_mlops.train import build_pipeline, candidate_classifiers


def test_candidate_classifiers_include_onnx_friendly_models() -> None:
    classifiers = candidate_classifiers(random_state=42)

    assert set(classifiers) == {
        "multinomial_logistic_regression",
        "random_forest",
        "gradient_boosting",
    }
    assert isinstance(classifiers["multinomial_logistic_regression"], LogisticRegression)
    assert isinstance(classifiers["random_forest"], RandomForestClassifier)
    assert isinstance(classifiers["gradient_boosting"], GradientBoostingClassifier)


def test_build_pipeline_scales_only_logistic_regression() -> None:
    classifiers = candidate_classifiers(random_state=42)

    logistic_steps = build_pipeline(
        "multinomial_logistic_regression",
        classifiers["multinomial_logistic_regression"],
    ).named_steps
    random_forest_steps = build_pipeline(
        "random_forest",
        classifiers["random_forest"],
    ).named_steps

    assert "imputer" in logistic_steps
    assert "scaler" in logistic_steps
    assert "classifier" in logistic_steps
    assert "imputer" in random_forest_steps
    assert "scaler" not in random_forest_steps
    assert "classifier" in random_forest_steps
