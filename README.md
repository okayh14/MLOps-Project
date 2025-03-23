Code Coverage: [![codecov](https://codecov.io/gh/michellebinder/heart-disease-prediction/branch/testing/graph/badge.svg)](https://codecov.io/gh/michellebinder/heart-disease-prediction)

Frontend (gehosted bei Azure):  [Web-Anwendung](http://predictmyheart.westeurope.cloudapp.azure.com:8501/ )

# Getting Started

## Prerequisites

- Python 3.9 or higher
- Git
- pip

## Repository klonen

```bash
git clone https://github.com/okayh14/MLOps-Project.git
cd MLOps-Project
```
## Optional virtuelle Umgebung aufsetze

```bash
python3 -m venv env
```
**Aktivierung der virtuellen Umgebung**

_Unter Windows:_
```bash
env\Scripts\activate
```
_Unter macOS/Linux:_
```bash
source env/bin/activate
```
## Installation der Abhängigkeiten

```bash
#Upgrade pip
pip install --upgrade pip
```

```bash
pip install -r backend/data_service/requirements.txt
pip install -r backend/model_training/requirements.txt
pip install -r backend/orchestrator/requirements.txt
```
## Installation der Abhängigkeiten fürs Testing
```
pip install -r test/requirements.txt
```
### Durchführung der Tests
```bash
pytest --cov=backend test/
```

## Starten der Services
```bash
# Start Data Service
uvicorn backend.data_service.api.api:app --reload --port 8001

# Start Model Training Service
uvicorn backend.model_training.app:app --reload --port 8002

# Start Orchestrator
uvicorn backend.Orchestrator.Orchestrator:app --reload --port 8000
```

