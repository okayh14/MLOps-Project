import warnings

import datetime
import pandas as pd
import numpy as np
import json

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Scikit-Learn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
    StandardScaler,
    RobustScaler,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.base import BaseEstimator, TransformerMixin
# Modelle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

client = MlflowClient()

def training_and_logging(json_data):
    """
    Liest den Datensatz ein und führt ein Pipeline-Experiment durch:
      - Modelle: LogisticRegression, RandomForest, XGBClassifier
      (LightGBM entfernt)
      - Erweitertes Grid je Modell
      - Encoder: OneHot, Ordinal, Label
      - Scaler: None, Standard, Robust (MinMax entfernt)
      - Bestimmte (OneHot, Standard)/(OneHot, Robust)-Kombis verboten
      - Feature Selection: None, SelectKBest(k=5)
      - 5-fold CV, Metriken = Accuracy, Precision, Recall, F1, F-Beta(1.5)
      - Top 4 Modelle werden in MLflow Model Registry registriert
    """

    ##########################
    # 1) Daten laden
    ##########################
    try:
        df = pd.DataFrame(json_data)
    except Exception as e:
        print(f"Error processing data: {e}")
        raise ValueError("Invalid JSON data provided.")
    
    target_col = "heart_attack_risk"

    # Falls es eine ID-Spalte gibt, entfernen
    if "PatientID" in df.columns:
        df.drop(columns=["PatientID"], inplace=True)

    # Features / Target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Numerische / kategorische Spalten
    # numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    ##########################
    # 2) MLflow-Experiment anlegen
    ##########################
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"heart_attack_experiment_{timestamp}"
    mlflow.set_experiment(experiment_name)

    ##########################
    # 3) Encoder-Optionen
    ##########################
    class PandasLabelEncoder:
        """LabelEncoder spaltenweise auf cat_cols."""

        def __init__(self, columns):
            self.columns = columns
            self.encoders = {}

        def fit(self, X, y=None):
            for col in self.columns:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.encoders[col] = le
            return self

        def transform(self, X):
            X_ = X.copy()
            for col in self.columns:
                X_[col] = self.encoders[col].transform(X_[col].astype(str))
            return X_

        def fit_transform(self, X, y=None):
            return self.fit(X, y).transform(X)

    encoder_options = {
        "OneHot": ("onehot", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        "Ordinal": ("ordinal", OrdinalEncoder(), cat_cols),
        "Label": ("label", PandasLabelEncoder(cat_cols), cat_cols),
    }

    ##########################
    # 4) Scaler-Optionen (MinMax entfernt)
    ##########################
    scaler_options = {
        "None": None,
        "Standard": StandardScaler(),
        "Robust": RobustScaler(),
    }

    ##########################
    # 5) Verbotene (Encoder, Scaler)-Kombos
    #    (OneHot + Standard/Robust => Sparse + Centering => Fehler)
    ##########################
    forbidden_encoder_scaler = {("OneHot", "Standard"), ("OneHot", "Robust")}

    ##########################
    # 6) Feature Selection
    ##########################
    feat_select_options = {
        "None": None,
        "SelectKBest": SelectKBest(score_func=f_classif, k=5),
    }

    ##########################
    # 7) Modell + ERWEITERTES Hyperparam-Grid
    #    (LightGBM weglassen, stattdessen mehr Kombis für LR/RF/XGB)
    ##########################
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

    ##########################
    # 8) Custom F-Beta=1.5 Scorer
    ##########################

    fbeta_1_5_scorer = make_scorer(fbeta_score, beta=1.5)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "fbeta_1_5": fbeta_1_5_scorer,
    }

    ##########################
    # 9) Cross-Validation Setup
    ##########################
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    ##########################
    # 10) Schleifen
    ##########################
    for encoder_name, (encoder_id, encoder_obj, enc_cols) in encoder_options.items():
        for scaler_name, scaler_obj in scaler_options.items():

            # Check: Ist diese Kombi verboten?
            if (encoder_name, scaler_name) in forbidden_encoder_scaler:
                print(
                    f"Skipping (Encoder={encoder_name}, Scaler={scaler_name}) - incompatible."
                )
                continue

            for fs_name, fs_obj in feat_select_options.items():
                for model_name, param_list in model_param_grid.items():
                    for param_idx, params in enumerate(param_list, start=1):

                        run_name = (
                            f"{model_name}_"
                            f"enc={encoder_name}_"
                            f"sc={scaler_name}_"
                            f"fs={fs_name}_"
                            f"paramset_{param_idx}"
                        )

                        with mlflow.start_run(run_name=run_name):
                            # Modell-Instanz
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
                                raise ValueError(f"Unbekanntes Modell: {model_name}")

                            # Pipeline-Schritte
                            pipeline_steps = []

                            # Encoder
                            if encoder_id in ("onehot", "ordinal"):
                                ct_transformers = []
                                if len(cat_cols) > 0:
                                    ct_transformers.append(
                                        (encoder_id, encoder_obj, cat_cols)
                                    )
                                column_transformer = ColumnTransformer(
                                    ct_transformers, remainder="passthrough"
                                )
                                pipeline_steps.append(
                                    ("column_transformer", column_transformer)
                                )
                            elif encoder_id == "label":
                                pipeline_steps.append(("label_encoder", encoder_obj))

                            # Scaler
                            if scaler_obj is not None:
                                pipeline_steps.append(
                                    (f"scaler_{scaler_name}", scaler_obj)
                                )

                            # Feature Selection
                            if fs_obj is not None:
                                pipeline_steps.append((fs_name, fs_obj))

                            # Classifier
                            pipeline_steps.append(("clf", clf))

                            pipeline = Pipeline(steps=pipeline_steps)

                            # Params loggen
                            mlflow.log_param("encoder", encoder_name)
                            mlflow.log_param("scaler", scaler_name)
                            mlflow.log_param("feature_selection", fs_name)
                            mlflow.log_param("model", model_name)
                            mlflow.log_param("param_set_idx", param_idx)
                            for k, v in params.items():
                                mlflow.log_param(k, v)

                            # Cross-Validation
                            cv_scores = cross_validate(
                                pipeline,
                                X,
                                y,
                                cv=cv,
                                scoring=scoring,
                                return_estimator=False,
                            )

                            # Mittelwerte berechnen
                            metrics_dict = {}
                            for score_name in scoring.keys():
                                metrics_dict[score_name] = np.mean(
                                    cv_scores[f"test_{score_name}"]
                                )

                            # Metriken loggen
                            mlflow.log_metrics(metrics_dict)

                            # Modell loggen
                            mlflow.sklearn.log_model(pipeline, artifact_path="model")

                            # Ergebnisse zwischenspeichern
                            run_id = mlflow.active_run().info.run_id
                            row = {
                                "run_id": run_id,
                                "run_name": run_name,
                                "encoder": encoder_name,
                                "scaler": scaler_name,
                                "feat_selection": fs_name,
                                "model_name": model_name,
                                "param_set_idx": param_idx,
                            }
                            row.update(params)
                            row.update(metrics_dict)
                            results.append(row)

                            mlflow.end_run()

    ##########################
    # 11) Top 4 nach F-Beta=1.5 registrieren
    ##########################
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.sort_values(by="fbeta_1_5", ascending=False, inplace=True)
        top5 = results_df.head(5)

        for rank, (idx, row) in enumerate(top5.iterrows(), start=1):
            best_run_id = row["run_id"]
            best_fbeta = row["fbeta_1_5"]
            best_model_name = row["model_name"]

            model_name_for_registry = f"{best_model_name}_{experiment_name}_Rank_{rank}"

            model_version = mlflow.register_model(
                model_uri=f"runs:/{best_run_id}/model", name=model_name_for_registry
            )

            client.transition_model_version_stage(
                name=model_name_for_registry,
                version=model_version.version,
                stage="Staging",
                archive_existing_versions=False,
            )

            client.update_model_version(
                name=model_name_for_registry,
                version=model_version.version,
                description=(
                    f"F-Beta(1.5) Score: {best_fbeta:.4f}\n"
                    f"Encoder: {row['encoder']}, Scaler: {row['scaler']}\n"
                    f"FeatSel: {row['feat_selection']}\n"
                    f"Model: {row['model_name']}, Params: {params}\n"
                ),
            )

        print("\n=== Training Completed ===")
        print(f"Experiment name: {experiment_name}")
        print(f"Total runs: {len(results_df)}")
        print("Top 4 by F-Beta(1.5):")
        print(
            top5[
                [
                    "run_id",
                    "model_name",
                    "encoder",
                    "scaler",
                    "feat_selection",
                    "fbeta_1_5",
                ]
            ]
        )
    else:
        print("No valid runs were executed (all combos were skipped or errored out).")
