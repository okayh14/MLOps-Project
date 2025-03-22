# Import necessary libraries
import warnings
import datetime
import os
import pandas as pd
import numpy as np
import json
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
    StandardScaler,
    RobustScaler,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from joblib import Parallel, delayed

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
# Initialize MLflow client for tracking experiments
client = MlflowClient()


def setup_experiment():
    """
    Creates a new MLflow experiment with a timestamp-based name.
    
    Returns:
        str: Name of the created experiment
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"heart_attack_experiment_{timestamp}"
    mlflow.set_experiment(experiment_name)
    return experiment_name


class LabelEncoderWrapper(BaseEstimator, TransformerMixin):
    """
    Custom wrapper for sklearn's LabelEncoder that handles unknown categories.
    
    This transformer stores encoders for each column and handles unknown values
    by adding an "Unknown" category during transformation.
    """

    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        """
        Fits a LabelEncoder for each column in the dataframe.
        
        Args:
            X: DataFrame with categorical columns to encode
            y: Target variable (not used, included for scikit-learn API compatibility)
            
        Returns:
            self: The fitted encoder
        """
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders[col] = le
        return self

    def transform(self, X):
        """
        Transforms categorical columns using the fitted encoders.
        
        Handles unknown values by mapping them to a dedicated "Unknown" category.
        
        Args:
            X: DataFrame with categorical columns to transform
            
        Returns:
            DataFrame: Transformed data with encoded categories
        """
        X_transformed = X.copy()
        for col in X.columns:
            le = self.encoders.get(col)
            if le:
                # Replace values not seen during training with "Unknown"
                X_transformed[col] = (
                    X_transformed[col]
                    .astype(str)
                    .apply(lambda val: val if val in le.classes_ else "Unknown")
                )
                # Add "Unknown" to classes if not already present
                if "Unknown" not in le.classes_:
                    le.classes_ = np.append(le.classes_, "Unknown")
                X_transformed[col] = le.transform(X_transformed[col].astype(str))
        return X_transformed


def configure_models(cat_cols):
    """
    Configures all model components and hyperparameter combinations for the experiment.
    
    Args:
        cat_cols: List of categorical column names
        
    Returns:
        tuple: Contains dictionaries for encoders, scalers, forbidden combinations,
               model parameters, feature selection options, and scoring metrics
    """
    # Define encoding options for categorical features
    encoder_options = {
        "OneHot": ("onehot", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        "Ordinal": (
            "ordinal",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            cat_cols,
        ),
        "Label": ("label", LabelEncoderWrapper(), cat_cols),
    }

    # Define scaling options for features
    scaler_options = {
        "None": None,
        "Standard": StandardScaler(),
        "Robust": RobustScaler(),
    }

    # Define incompatible combinations of encoders and scalers
    # OneHot encoding produces sparse matrices which are not compatible with scalers
    forbidden_combos = {("OneHot", "Standard"), ("OneHot", "Robust")}

    # Define model types and their hyperparameter combinations to try
    model_param_grid = {
        "LogisticRegression": [
            {"C": 0.01, "max_iter": 300, "solver": "lbfgs"},
            {"C": 0.1, "max_iter": 300, "solver": "lbfgs"},
            {"C": 1.0, "max_iter": 500, "solver": "lbfgs"},
            {"C": 10.0, "max_iter": 500, "solver": "lbfgs"},
        ],
        "RandomForest": [
            {"n_estimators": 50, "max_depth": 5},
            {"n_estimators": 100, "max_depth": 10},
            {"n_estimators": 200, "max_depth": 15},
            {"n_estimators": 300, "max_depth": None},
        ],
        "XGBClassifier": [
            {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3},
            {"n_estimators": 100, "learning_rate": 0.01, "max_depth": 5},
            {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 7},
            {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 10},
        ],
    }

    # Define feature selection options
    feat_select_options = {
        "None": None,
        "SelectKBest": SelectKBest(score_func=f_classif, k=5),
    }

    # Define scoring metrics with custom fbeta metric that favors recall (beta=1.5)
    fbeta_1_5_scorer = make_scorer(fbeta_score, beta=1.5)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "fbeta_1_5": fbeta_1_5_scorer,
    }

    return (
        encoder_options,
        scaler_options,
        forbidden_combos,
        model_param_grid,
        feat_select_options,
        scoring,
    )


def train_and_evaluate(
    X,
    y,
    encoder_options,
    scaler_options,
    forbidden_combos,
    model_param_grid,
    feat_select_options,
    scoring,
    experiment_name,
    n_jobs=-1,
):
    """
    Train models using cross-validation and evaluate them with multiprocessing support.
    
    Args:
        X: Feature dataframe
        y: Target variable
        encoder_options: Dictionary of encoding options
        scaler_options: Dictionary of scaling options
        forbidden_combos: Set of incompatible encoder-scaler combinations
        model_param_grid: Dictionary of models and their hyperparameter combinations
        feat_select_options: Dictionary of feature selection options
        scoring: Dictionary of scoring metrics
        experiment_name: Name of the MLflow experiment
        n_jobs: Number of parallel jobs to run (-1 for all available cores)
        
    Returns:
        DataFrame: Results of all model combinations with their metrics
    """
    # Configure cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    # Create list of all possible model configurations to try
    tasks = []
    for encoder_name, (encoder_id, encoder_obj, enc_cols) in encoder_options.items():
        for scaler_name, scaler_obj in scaler_options.items():
            # Skip incompatible combinations
            if (encoder_name, scaler_name) in forbidden_combos:
                print(
                    f"Skipping incompatible combination: Encoder={encoder_name}, Scaler={scaler_name}"
                )
                continue
            for fs_name, fs_obj in feat_select_options.items():
                for model_name, param_list in model_param_grid.items():
                    for param_idx, params in enumerate(param_list, start=1):
                        tasks.append(
                            (
                                encoder_name,
                                encoder_id,
                                encoder_obj,
                                enc_cols,
                                scaler_name,
                                scaler_obj,
                                fs_name,
                                fs_obj,
                                model_name,
                                params,
                                param_idx,
                            )
                        )

    # Execute all tasks in parallel using Joblib
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_task)(X, y, task, cv, scoring, experiment_name)
        for task in tasks
    )

    # Return results as a dataframe, filtering out any failed runs
    return pd.DataFrame([r for r in results if r is not None])


def process_task(X, y, task, cv, scoring, experiment_name):
    """
    Process a single model training and evaluation task.
    
    Args:
        X: Feature dataframe
        y: Target variable
        task: Tuple containing all parameters for this task
        cv: Cross-validation strategy
        scoring: Dictionary of scoring metrics
        experiment_name: Name of the MLflow experiment
        
    Returns:
        dict: Results for this model configuration or None if an error occurred
    """
    # Unpack the task parameters
    (
        encoder_name,
        encoder_id,
        encoder_obj,
        enc_cols,
        scaler_name,
        scaler_obj,
        fs_name,
        fs_obj,
        model_name,
        params,
        param_idx,
    ) = task

    # Create a descriptive run name for MLflow tracking
    run_name = f"{model_name}_enc={encoder_name}_sc={scaler_name}_fs={fs_name}_paramset_{param_idx}"

    try:
        # Start a new MLflow run for tracking this model configuration
        with mlflow.start_run(run_name=run_name):
            # Create directory for serialized models and save feature names
            serialized_models_dir = "./serialized_models"
            features_path = os.path.join(serialized_models_dir, "model_features.json")
            if not os.path.exists(features_path):
                with open(features_path, "w") as f:
                    json.dump({"columns": list(X.columns)}, f)

            # Create the ML pipeline with selected components
            pipeline_steps = []

            # Add appropriate encoder to the pipeline
            if encoder_id in ("onehot", "ordinal"):
                ct_transformers = []
                if len(enc_cols) > 0:
                    ct_transformers.append((encoder_id, encoder_obj, enc_cols))
                column_transformer = ColumnTransformer(
                    ct_transformers, remainder="passthrough"
                )
                pipeline_steps.append(("column_transformer", column_transformer))
            elif encoder_id == "label":
                pipeline_steps.append(("label_encoder", encoder_obj))

            # Add scaler to the pipeline if specified
            if scaler_obj is not None:
                pipeline_steps.append((f"scaler_{scaler_name}", scaler_obj))

            # Add feature selection to the pipeline if specified
            if fs_obj is not None:
                pipeline_steps.append((fs_name, fs_obj))

            # Add the classifier to the pipeline with specified hyperparameters
            if model_name == "LogisticRegression":
                clf = LogisticRegression(**params, random_state=42)
            elif model_name == "RandomForest":
                clf = RandomForestClassifier(**params, random_state=42)
            elif model_name == "XGBClassifier":
                clf = XGBClassifier(
                    **params,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=42,
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")

            pipeline_steps.append(("clf", clf))
            pipeline = Pipeline(steps=pipeline_steps)

            # Log all parameters to MLflow
            mlflow.log_param("encoder", encoder_name)
            mlflow.log_param("scaler", scaler_name)
            mlflow.log_param("feature_selection", fs_name)
            mlflow.log_param("model", model_name)
            mlflow.log_param("param_set_idx", param_idx)
            for k, v in params.items():
                mlflow.log_param(k, v)

            # Perform cross-validation and save each fold's model
            cv_scores = cross_validate(
                pipeline, X, y, cv=cv, scoring=scoring, return_estimator=True
            )

            # Calculate average metrics across all folds
            metrics_dict = {}
            for score_name in scoring.keys():
                metrics_dict[score_name] = np.mean(cv_scores[f"test_{score_name}"])

            # Log metrics to MLflow
            mlflow.log_metrics(metrics_dict)

            # Save the model from first fold as an artifact
            best_model = cv_scores["estimator"][0]  # Or choose based on performance
            mlflow.sklearn.log_model(best_model, artifact_path="model")

            # Compile results for the results dataframe
            run_id = mlflow.active_run().info.run_id
            result = {
                "run_id": run_id,
                "run_name": run_name,
                "encoder": encoder_name,
                "scaler": scaler_name,
                "feat_selection": fs_name,
                "model_name": model_name,
                "param_set_idx": param_idx,
            }
            result.update(params)
            result.update(metrics_dict)

            # End the MLflow run
            mlflow.end_run()
            return result

    except Exception as e:
        # Handle and log any errors
        print(f"Error in run {run_name}: {e}")
        if mlflow.active_run():
            mlflow.end_run()
        return None


async def main(json_data):
    """
    Main function to orchestrate the entire training and model registration pipeline.
    
    Args:
        json_data: JSON data containing features and target variable
        
    Returns:
        dict: Status and summary of the training process
    """
    try:
        # Setup experiment
        experiment_name = setup_experiment()
        print(f"Created experiment: {experiment_name}")

        # Process data
        df = pd.DataFrame(json_data)
        target_col = "heart_attack_risk"

        # Remove ID column if present
        if "PatientID" in df.columns:
            df.drop(columns=["PatientID"], inplace=True)

        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Create directory for serialized models and save feature names
        serialized_models_dir = "./serialized_models"
        os.makedirs(serialized_models_dir, exist_ok=True)
        features_path = os.path.join(serialized_models_dir, "model_features.json")
        with open(features_path, "w") as f:
            json.dump({"columns": list(X.columns)}, f)

        # Log dataset statistics
        print(
            f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(cat_cols)} categorical features"
        )

        # Configure all model components and hyperparameters
        (
            encoder_options,
            scaler_options,
            forbidden_combos,
            model_param_grid,
            feat_select_options,
            scoring,
        ) = configure_models(cat_cols)

        # Train and evaluate all model combinations
        print("Starting training pipeline...")
        results_df = train_and_evaluate(
            X,
            y,
            encoder_options,
            scaler_options,
            forbidden_combos,
            model_param_grid,
            feat_select_options,
            scoring,
            experiment_name,
        )

        # Commented out model registration code
        # register_top_models(results_df, experiment_name, top_n=5)

        # Print summary of training process
        print("\n=== Training Completed ===")
        print(f"Experiment name: {experiment_name}")

        # Return results
        return {
            "status": "success",
            "experiment_name": experiment_name,
            "total_runs": len(results_df),
            "results_df": results_df,
        }

    except Exception as e:
        # Handle and log any errors
        print(f"Error in training pipeline: {e}")
        return {"status": "error", "message": str(e)}
