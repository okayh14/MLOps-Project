# Use Python 3.9 base image
FROM python:3.9

# Set the working directory to the Orchestrator module
WORKDIR /app/backend/Orchestrator

# Copy requirements and install Python dependencies without caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH so "from backend.xyz" works
ENV PYTHONPATH=/app

# Ensure the application is treated as a Python package (needed for module imports)
RUN touch /app/__init__.py

# Expose port 8003 so the app is accessible outside the container
EXPOSE 8003

# Command to run the FastAPI app using Uvicorn with hot-reload enabled
CMD ["uvicorn", "orchestrator:app", "--host", "0.0.0.0", "--port", "8003", "--reload"]
