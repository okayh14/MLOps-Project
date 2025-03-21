FROM python:3.9

WORKDIR /app

# Kopiere Requirements und installiere sie
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere den gesamten Code ins Image
COPY . .

# Exponiere den Port 
EXPOSE 8002

# Starte FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]
