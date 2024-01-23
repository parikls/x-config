import os
from pathlib import Path

from dotenv import load_dotenv

from x_config import XConfigError

SECRETS_DOTENV_NAME = 'secrets_dotenv_name'


def load_dotenv_secrets(dotenv_dir: Path, config: dict) -> dict:
    """
    Loads secrets from dot-env file and returns them in a dict format
    """
    try:
        dotenv_name = config.pop(SECRETS_DOTENV_NAME)
    except KeyError:
        raise XConfigError(f'unable to extract `{SECRETS_DOTENV_NAME} property')
    load_dotenv(dotenv_dir / dotenv_name)
    return {k.upper(): v for k, v in os.environ.items()}
