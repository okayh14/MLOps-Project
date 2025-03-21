import os
import sys
from pathlib import Path

# Setze die Umgebungsvariable vor allen Tests
def pytest_configure(config):
    os.environ["DATABASE_URL"] = "sqlite:///test_db.sqlite"

# Füge den Hauptprojektpfad zum sys.path hinzu, falls nötig
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))