import zipfile
import mlflow
from mlflow.tracking import MlflowClient
import dill
import os
from joblib import dump
import joblib

client = MlflowClient()


async def register_top_models(results_df, experiment_name, top_n=5):
    """
    Register the top N models based on F-Beta score to the model registry
    """
    if results_df.empty:
        print("No valid runs to register")
        return

    # Sort by F-Beta score
    results_df.sort_values(by="fbeta_1_5", ascending=False, inplace=True)
    top_models = results_df.head(top_n)

    print(f"\n=== Registering Top {top_n} Models ===")
    print(f"Total runs: {len(results_df)}")
    print(f"Top {top_n} by F-Beta(1.5):")
    print(
        top_models[
            ["run_id", "model_name", "encoder", "scaler", "feat_selection", "fbeta_1_5"]
        ]
    )

    for rank, (idx, row) in enumerate(top_models.iterrows(), start=1):
        run_id = row["run_id"]
        fbeta_score = row["fbeta_1_5"]
        model_name = row["model_name"]

        # Extract params for description
        params = {
            k: v
            for k, v in row.items()
            if k
            in ["C", "max_iter", "solver", "n_estimators", "max_depth", "learning_rate"]
        }

        # Create a unique name for registry
        model_name_for_registry = f"{model_name}_{experiment_name}_Rank_{rank}"

        # Register the model
        try:
            model_version = mlflow.register_model(
                model_uri=f"runs:/{run_id}/model", name=model_name_for_registry
            )

            # Transition to staging
            client.transition_model_version_stage(
                name=model_name_for_registry,
                version=model_version.version,
                stage="Staging",
                archive_existing_versions=False,
            )

            # Update description
            client.update_model_version(
                name=model_name_for_registry,
                version=model_version.version,
                description=(
                    f"F-Beta(1.5) Score: {fbeta_score:.4f}\n"
                    f"Encoder: {row['encoder']}, Scaler: {row['scaler']}\n"
                    f"FeatSel: {row['feat_selection']}\n"
                    f"Model: {row['model_name']}, Params: {params}\n"
                ),
            )

            print(
                f"Registered model: {model_name_for_registry} (Version: {model_version.version})"
            )

        except Exception as e:
            print(f"Failed to register model: {e}")


async def serialize_and_compress_models(serialized_directory):
    try:
        models = client.search_registered_models()

        # Ensure directory exists
        os.makedirs(serialized_directory, exist_ok=True)

        for model in models:
            for version in model.latest_versions:
                model_uri = f"models:/{model.name}/{version.version}"
                model_path = os.path.join(
                    serialized_directory, f"{model.name}_v{version.version}.pkl"
                )
                model = mlflow.sklearn.load_model(model_uri)
                joblib.dump(model, model_path)  # Use joblib to dump the model
    except Exception as e:
        print(f"Failed to serialize models with Joblib: {e}")
        raise


async def clean_model_registry_and_folder(folder_path=None):
    """
    Empty the MLflow model registry and optionally remove all files in a specified folder.
    
    Parameters:
    folder_path (str, optional): Path to the folder to clean. If None, only the model registry is cleaned.
    
    Returns:
    tuple: (num_models_deleted, num_files_deleted)
    """
    client = MlflowClient()
    models_deleted = 0
    files_deleted = 0

    # Clean the model registry
    try:
        # Get all registered models
        registered_models = client.search_registered_models()

        if registered_models:
            print(f"Found {len(registered_models)} registered models to delete.")

            for model in registered_models:
                model_name = model.name

                # Delete all versions of the model
                versions = client.get_latest_versions(model_name)
                for version in versions:
                    try:
                        # If the version is in production or staging, transition it to archived first
                        if version.current_stage in ["Production", "Staging"]:
                            client.transition_model_version_stage(
                                name=model_name,
                                version=version.version,
                                stage="Archived",
                            )

                        # Delete the version
                        client.delete_model_version(
                            name=model_name, version=version.version
                        )
                        print(
                            f"Deleted model version {version.version} of {model_name}"
                        )
                    except Exception as e:
                        print(
                            f"Error deleting model version {version.version} of {model_name}: {e}"
                        )

                # Delete the registered model
                try:
                    client.delete_registered_model(model_name)
                    print(f"Deleted registered model: {model_name}")
                    models_deleted += 1
                except Exception as e:
                    print(f"Error deleting registered model {model_name}: {e}")
        else:
            print("No registered models found to delete.")
    except Exception as e:
        print(f"Error cleaning model registry: {e}")

    # Clean the specified folder if provided
    if folder_path:
        if os.path.exists(folder_path):
            try:
                # Count files before deletion
                file_count = sum([len(files) for _, _, files in os.walk(folder_path)])

                # Remove all files in the folder
                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                        files_deleted += 1
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        files_deleted += 1  # Counting directories as one item

                print(f"Cleaned folder: {folder_path} - Removed {files_deleted} items")
            except Exception as e:
                print(f"Error cleaning folder {folder_path}: {e}")
        else:
            print(f"Folder {folder_path} does not exist.")

    return models_deleted, files_deleted
