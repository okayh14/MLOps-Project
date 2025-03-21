FROM python:3.9

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Make sure Orchestrator is a package (create __init__.py)
RUN touch /app/__init__.py

EXPOSE 8003

# Start orchestrator
CMD ["uvicorn", "orchestrator:app", "--host", "0.0.0.0", "--port", "8003", "--reload"]
