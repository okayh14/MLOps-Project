import os
import sys
from pathlib import Path

# Set the environment variable before all tests
def pytest_configure(config):
    os.environ["DATABASE_URL"] = "sqlite:///test_db.sqlite"

# Add the main project path to sys.path if needed
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))