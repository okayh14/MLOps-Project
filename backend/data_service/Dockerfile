# Use official Python base image
FROM python:3.9

# Set working directory inside container
WORKDIR /app/backend/data_service

# Copy and install dependencies
COPY backend/data_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add PYTHONPATH so "from backend.xyz" works
ENV PYTHONPATH=/app

# Create wait-for-db script to wait until Postgres is ready before starting app
RUN echo '#!/bin/bash\n\
echo "⏳ Waiting for PostgreSQL to become available..."\n\
until nc -z db 5432; do\n\
  sleep 1\n\
done\n\
echo "PostgreSQL is up! Starting application..."\n\
python init.py && uvicorn api:app --host 0.0.0.0 --port 8001 --reload\n\
' > /app/wait-for-db.sh && chmod +x /app/wait-for-db.sh

# Use the wait script as container entrypoint
CMD ["/bin/bash", "/app/wait-for-db.sh"]
