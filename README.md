## Virtual Environment Setup

### macOS/Linux

```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r backend/Data\ Service/requirements.txt
pip install -r backend/Model\ Training/requirements.txt
pip install -r backend/Orchestrator/requirements.txt
```

### windows

```bash
python3 -m venv env
env\Scripts\activate
pip install --upgrade pip
pip install -r backend/Data\ Service/requirements.txt
pip install -r backend/Model\ Training/requirements.txt
pip install -r backend/Orchestrator/requirements.txt
```
