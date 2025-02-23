import os
from dotenv import load_dotenv

load_dotenv()

def load_env(key):
    """Load environment variables securely."""
    return os.getenv(key)
