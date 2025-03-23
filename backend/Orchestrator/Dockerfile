# Use Python 3.9 base image
FROM python:3.9

# Set the working directory to the Orchestrator module
WORKDIR /app/backend/Orchestrator

# Copy requirements and install dependencies
COPY backend/Orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole project into the container
COPY . .

# Set PYTHONPATH so "from backend.xyz" imports work
ENV PYTHONPATH=/app

# Expose port 8003 so the app is accessible outside the container
EXPOSE 8003

# Run the FastAPI app with hot-reload
CMD ["uvicorn", "orchestrator:app", "--host", "0.0.0.0", "--port", "8003", "--reload"]
