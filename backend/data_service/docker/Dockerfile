# Basierend auf Python 3.9
FROM python:3.9

# Setze das Arbeitsverzeichnis auf /app im Container
WORKDIR /app

# Kopiere die requirements.txt Datei aus dem app-Verzeichnis
COPY ../requirements.txt .

# Installiere alle benötigten Python-Pakete
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere den gesamten Inhalt des übergeordneten Projektordners in das /app Verzeichnis im Container
COPY ../ .

# Starte den Uvicorn Server mit FastAPI
CMD ["python", "scripts/init.py", "&&", "uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
