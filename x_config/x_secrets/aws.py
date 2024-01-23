import json

import boto3

SECRETS_AWS_SECRET_NAME = 'secrets_aws_secret_name'


def load_aws_secrets(secret_name: str, region: str) -> dict:
    """
    Loads secrets from AWS Secrets and returns them in a dict format
    """
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region)
    get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    contents = json.loads(get_secret_value_response['SecretString'])
    return {k.upper(): v for k, v in contents.items()}
