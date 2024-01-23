import json

import boto3

from x_config import XConfigError

SECRETS_AWS_SECRET_NAME = 'secrets_aws_secret_name'
SECRETS_AWS_REGION = 'secrets_aws_region'


def load_aws_secrets(config: dict) -> dict:
    """
    Loads secrets from AWS Secrets and returns them in a dict format
    """
    try:
        secret_name = config.pop(SECRETS_AWS_SECRET_NAME)
    except KeyError:
        raise XConfigError(f'unable to extract `{SECRETS_AWS_SECRET_NAME} property')
    try:
        region = config.pop(SECRETS_AWS_REGION)
    except KeyError:
        raise XConfigError(f'unable to extract `{SECRETS_AWS_REGION} property')

    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region)
    get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    contents = json.loads(get_secret_value_response['SecretString'])
    return {k.upper(): v for k, v in contents.items()}
