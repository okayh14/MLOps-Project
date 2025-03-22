from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# --- Database configuration ---
# This is the connection string to your PostgreSQL database.
# Format: postgresql://<username>:<password>@<host>:<port>/<database>
DATABASE_URL = "postgresql://sami:secret@db:5432/heart_risk_data"

# Create a SQLAlchemy Engine that manages the connection pool
# 'db' refers to the Docker service name of the PostgreSQL container
engine = create_engine(DATABASE_URL)

# Create a configured "Session" class
# - autoflush=False: disables automatic flushes (write to DB only when needed)
# - autocommit=False: disables autocommit; changes are only saved on explicit commit
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Base class for all ORM models (e.g. PatientData)
# Used as a foundation for declaring table models
Base = declarative_base()
