import pytest
import pandas as pd
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from backend.model_training.inference import prepare_and_predict

@pytest.mark.asyncio
async def test_inference_end_to_end(tmp_path):
    # === Step 1: Train binary model ===
    iris = load_iris()
    X, y = iris.data, iris.target
    # Mach das Target binär (z. B. Klasse 1 vs. Rest)
    binary_y = (y == 1).astype(int)

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)

    model = LogisticRegression(max_iter=1000)
    model.fit(df, binary_y)

    # === Step 2: Save model ===
    model_path = tmp_path / "MyModel_v1.pkl"
    joblib.dump(model, model_path)

    # === Step 3: Save model_features.json ===
    with open(tmp_path / "model_features.json", "w") as f:
        json.dump({"columns": feature_names}, f)

    # === Step 4: Prepare input data (as in real prediction request) ===
    input_features = df.head(5).to_dict(orient="records")  # simulate 5 patients

    # === Step 5: Run real prediction ===
    result = prepare_and_predict(input_features, str(tmp_path))

    # === Step 6: Assert results ===
    assert isinstance(result, pd.DataFrame)
    assert "Probability_At_Risk" in result.columns
    assert "Classifier" in result.columns
    assert len(result) == 5
    assert result["Probability_At_Risk"].between(0, 1).all()
