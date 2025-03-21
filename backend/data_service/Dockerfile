FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Warte-Script anlegen (z. B. wait-for-db.sh)
RUN echo '#!/bin/bash\n\
echo "Waiting for Postgres..."\n\
until nc -z db 5432; do\n\
  sleep 1\n\
done\n\
echo "Postgres is up!"\n\
python init.py && uvicorn api:app --host 0.0.0.0 --port 8001 --reload\n\
' > /app/wait-for-db.sh && chmod +x /app/wait-for-db.sh

CMD ["/bin/bash", "/app/wait-for-db.sh"]
