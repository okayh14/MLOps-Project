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

nacheinander ausführen:

- 1. docker compose build --no-cache
- 2. docker compose up

nur wenn man wieder stoppen will:
- 3. docker compose down --volumes --remove-orphans
