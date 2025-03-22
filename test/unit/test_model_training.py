import pytest
import pandas as pd
import numpy as np
import mlflow
import os
import logging
from unittest.mock import patch, MagicMock
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from unittest.mock import patch, MagicMock
from backend.model_training.model_training import (
    setup_experiment,
    LabelEncoderWrapper,
    configure_models,
    train_and_evaluate,
)


@pytest.fixture
def mock_data():
    """
    Creates a mock dataset for testing model training functions.
    Includes categorical and numerical features.
    """
    return pd.DataFrame(
        {
            "age": [45, 56, 65, 50, 52, 60, 48, 55, 62, 58],
            "sex": ["Male", "Female"] * 5,
            "cholesterol": [220, 195, 250, 210, 200, 240, 215, 230, 225, 205],
            "systolic_blood_pressure": [
                120,
                130,
                140,
                125,
                135,
                128,
                130,
                132,
                138,
                140,
            ],
            "diastolic_blood_pressure": [80, 85, 90, 80, 85, 88, 82, 84, 86, 88],
            "heart_rate": [72, 80, 76, 75, 78, 70, 74, 73, 77, 79],
            "diabetes": [False, True] * 5,
            "family_history": [True, False] * 5,
            "smoking": [False, False, True, True, False] * 2,
            "obesity": [False, True] * 5,
            "alcohol_consumption": [False, True] * 5,
            "exercise_hours_per_week": [
                3.5,
                2.0,
                4.0,
                1.5,
                2.5,
                3.0,
                4.5,
                1.0,
                2.2,
                3.3,
            ],
            "diet": [
                "Average",
                "Poor",
                "Healthy",
                "Average",
                "Poor",
                "Healthy",
                "Average",
                "Poor",
                "Healthy",
                "Average",
            ],
            "previous_heart_problems": [False, True] * 5,
            "medication_use": [False, True] * 5,
            "stress_level": [5, 7, 6, 8, 5, 7, 6, 8, 5, 7],
            "sedentary_hours_per_day": [
                6.2,
                7.5,
                5.0,
                8.0,
                6.0,
                7.2,
                5.5,
                8.1,
                6.3,
                7.0,
            ],
            "income": [
                50000,
                60000,
                55000,
                62000,
                58000,
                59000,
                51000,
                63000,
                54000,
                56000,
            ],
            "bmi": [28.5, 32.0, 26.5, 30.2, 29.0, 31.1, 27.3, 33.0, 28.1, 30.0],
            "triglycerides": [150, 180, 160, 170, 155, 165, 158, 172, 159, 161],
            "physical_activity_days_per_week": [3, 2, 4, 1, 3, 2, 4, 1, 3, 2],
            "sleep_hours_per_day": [7, 6, 8, 5, 7, 6, 8, 5, 7, 6],
            "country": ["Germany"] * 10,
            "continent": ["Europe"] * 10,
            "hemisphere": ["Northern"] * 10,
            "heart_attack_risk": [0, 1] * 5,
        }
    )


def test_setup_experiment():
    """
    Verifies that a new MLflow experiment is created and returns a valid name.
    """
    with patch("mlflow.set_experiment") as mock_set_experiment:
        experiment_name = setup_experiment()
        assert "heart_attack_experiment_" in experiment_name
        mock_set_experiment.assert_called_once()


def test_label_encoder_wrapper_dataframe(mock_data):
    """
    Tests that LabelEncoderWrapper correctly transforms categorical columns.
    """
    X = mock_data[["sex", "diet"]]
    encoder = LabelEncoderWrapper()
    encoder.fit(X)
    X_transformed = encoder.transform(X)

    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape == X.shape
    assert set(X_transformed.columns) == set(X.columns)
    assert all(
        pd.api.types.is_integer_dtype(X_transformed[col])
        for col in X_transformed.columns
    )


def test_label_encoder_wrapper_unknown_value():
    """
    Tests that LabelEncoderWrapper handles unknown categories gracefully.
    """
    import pandas as pd

    train = pd.DataFrame({"col": ["A", "B", "C"]})
    test = pd.DataFrame({"col": ["A", "B", "D"]})  # 'D' is unknown

    encoder = LabelEncoderWrapper()
    encoder.fit(train)
    transformed = encoder.transform(test)

    assert isinstance(transformed, pd.DataFrame)
    assert "col" in transformed.columns
    assert (
        transformed["col"]
        .isin(encoder.encoders["col"].transform(encoder.encoders["col"].classes_))
        .all()
    )


def test_configure_models():
    """
    Tests that model configuration options return dictionaries as expected.
    """
    cat_cols = ["sex"]
    (
        encoder_options,
        scaler_options,
        forbidden_combos,
        model_param_grid,
        feat_select_options,
        scoring,
    ) = configure_models(cat_cols)

    assert isinstance(encoder_options, dict)
    assert isinstance(scaler_options, dict)
    assert isinstance(model_param_grid, dict)
    assert isinstance(feat_select_options, dict)
    assert isinstance(scoring, dict)


def test_train_and_evaluate(mock_data, tmp_path):
    """
    Tests the full training and evaluation pipeline with minimal configuration.
    All MLflow artifacts and tracking data are stored in a temporary directory.
    
    Args:
        mock_data (pd.DataFrame): Synthetic test dataset provided by pytest fixture
        tmp_path (pathlib.Path): Temporary directory provided by pytest for isolated file operations
    """
    logging.getLogger("mlflow").setLevel(logging.ERROR)

    # Tell MLflow to use temp dir instead of creating ./mlruns
    mlflow.set_tracking_uri(tmp_path.as_uri())

    # Set working directory to temporary path
    os.chdir(tmp_path)
    # Create dummy directory expected by the pipeline
    (tmp_path / "serialized_models").mkdir(parents=True, exist_ok=True)

    # Prepare data
    X = mock_data.drop(columns=["heart_attack_risk"])
    y = mock_data["heart_attack_risk"]

    # Minimal config for two models
    encoder_options = {"Label": ("label", LabelEncoderWrapper(), ["sex"])}
    scaler_options = {"None": None}
    forbidden_combos = []
    model_param_grid = {
        "LogisticRegression": [{"C": 0.01, "solver": "lbfgs"}],
        "RandomForest": [{"n_estimators": 10, "max_depth": 3}],
    }
    feat_select_options = {"None": None}
    scoring = {"accuracy": "accuracy"}

    # Prepare mocked MLflow context
    fake_run = MagicMock()
    fake_run.__enter__.return_value = fake_run
    fake_run.info.run_id = "test_run_id"

    with patch("mlflow.start_run", return_value=fake_run), patch(
        "mlflow.active_run", return_value=fake_run
    ), patch("mlflow.log_param"), patch("mlflow.log_metrics"), patch(
        "mlflow.sklearn.log_model"
    ):

        results = train_and_evaluate(
            X,
            y,
            encoder_options,
            scaler_options,
            forbidden_combos,
            model_param_grid,
            feat_select_options,
            scoring,
            experiment_name="test_experiment",
            n_jobs=1,
        )

    assert isinstance(results, pd.DataFrame)
    assert len(results) >= 2


def test_forbidden_combo_skipped(mock_data):
    """
    Verifies that a forbidden combination of encoder and scaler is skipped.
    """
    X = mock_data.drop(columns=["heart_attack_risk"])
    y = mock_data["heart_attack_risk"]

    encoder_options = {
        "OneHot": ("onehot", OneHotEncoder(handle_unknown="ignore"), ["sex"])
    }
    scaler_options = {"Standard": StandardScaler()}
    forbidden_combos = [("OneHot", "Standard")]
    model_param_grid = {"LogisticRegression": [{"C": 1.0}]}
    feat_select_options = {"None": None}
    scoring = {"accuracy": "accuracy"}

    with patch("mlflow.start_run"), patch("mlflow.log_param"), patch(
        "mlflow.log_metrics"
    ):
        results = train_and_evaluate(
            X,
            y,
            encoder_options,
            scaler_options,
            forbidden_combos,
            model_param_grid,
            feat_select_options,
            scoring,
            experiment_name="test_experiment",
            n_jobs=1,
        )

    # Should skip the forbidden combination, resulting in no model runs
    assert results.empty


def test_invalid_model_name_handling(mock_data):
    """
    Ensures that unknown model names are handled gracefully and skipped.
    """
    X = mock_data.drop(columns=["heart_attack_risk"])
    y = mock_data["heart_attack_risk"]

    encoder_options = {"Label": ("label", LabelEncoderWrapper(), ["sex"])}
    scaler_options = {"None": None}
    forbidden_combos = []
    model_param_grid = {"NonExistentModel": [{"some_param": 123}]}
    feat_select_options = {"None": None}
    scoring = {"accuracy": "accuracy"}

    with patch("mlflow.start_run"), patch("mlflow.log_param"), patch(
        "mlflow.log_metrics"
    ):
        results = train_and_evaluate(
            X,
            y,
            encoder_options,
            scaler_options,
            forbidden_combos,
            model_param_grid,
            feat_select_options,
            scoring,
            experiment_name="test_experiment",
            n_jobs=1,
        )

    # Should be empty because the model is not supported
    assert results.empty
