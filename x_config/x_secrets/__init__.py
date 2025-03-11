from enum import Enum
from x_config.x_secrets.dotenv import SECRETS_DOTENV_NAME
from x_config.x_secrets.aws import SECRETS_AWS_SECRET_NAME, SECRETS_AWS_REGION
from x_config.x_secrets.kube import SECRETS_KUBE_NAMESPACE, SECRETS_KUBE_SECRET_NAME

SECRETS_SOURCE_PROP_NAME = 'secrets_source'


class SecretsSource(Enum):
    """
    Supported sources of secrets:

    - .env file
    - AWS Secrets service
    """
    DOTENV = 'dotenv'
    AWS = 'aws'
    KUBE = 'kube'


SECRET_SPECIFIC_PROPS = {
    SECRETS_DOTENV_NAME,
    SECRETS_AWS_SECRET_NAME,
    SECRETS_AWS_REGION,
    SECRETS_KUBE_SECRET_NAME,
    SECRETS_KUBE_NAMESPACE,
    SECRETS_SOURCE_PROP_NAME
}


SECRET_SOURCE_REQUIRED_PROPS = {
    SecretsSource.DOTENV: {SECRETS_DOTENV_NAME, },
    SecretsSource.AWS: {SECRETS_AWS_SECRET_NAME, SECRETS_AWS_REGION},
    SecretsSource.KUBE: {SECRETS_KUBE_SECRET_NAME, SECRETS_KUBE_NAMESPACE}
}
