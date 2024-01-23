import os
from pathlib import Path

from dotenv import load_dotenv

SECRETS_DOTENV_NAME = 'secrets_dotenv_name'


def load_dotenv_secrets(dotenv_dir: Path, dotenv_name: str) -> dict:
    """
    Loads secrets from dot-env file and returns them in a dict format
    """
    load_dotenv(dotenv_dir / dotenv_name)
    return {k.upper(): v for k, v in os.environ.items()}
