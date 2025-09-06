"""Configuration settings for the Telangana AI Analyst."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "output"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

# Application Configuration
APP_NAME = "RTGS"  # Real-Time Government System

# Flask Configuration
FLASK_ENV = os.getenv("FLASK_ENV", "development")
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

# CrewAI Configuration
CREWAI_VERBOSE = os.getenv("CREWAI_VERBOSE", "True").lower() == "true"
CREWAI_MEMORY = os.getenv("CREWAI_MEMORY", "True").lower() == "true"

# Data Processing Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.json'}

# Analysis Configuration
DEFAULT_CONFIDENCE_THRESHOLD = 0.8
MAX_INSIGHTS_PER_DATASET = 10
