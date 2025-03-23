# Use official Python base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app/backend/model_training

# Copy the requirements file and install dependencies
COPY backend/model_training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add PYTHONPATH so "from backend.xyz" works
ENV PYTHONPATH=/app

# Expose the port that the service will run on
EXPOSE 8002

# Start the FastAPI application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]
