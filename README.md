[![codecov](https://codecov.io/gh/okayh14/MLOps-Project/branch/main/graph/badge.svg)](https://codecov.io/gh/okayh14/MLOps-Project)

## Virtual Environment Setup

### macOS/Linux

```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r backend/data_service/requirements.txt
pip install -r backend/model_training/requirements.txt
pip install -r backend/orchestrator/requirements.txt
pip install -r test/requirements.txt
```

### Windows

```bash
python3 -m venv env
env\Scripts\activate
pip install --upgrade pip
pip install -r backend/data_service/requirements.txt
pip install -r backend/model_training/requirements.txt
pip install -r backend/orchestrator/requirements.txt
pip install -r test/requirements.txt
```
