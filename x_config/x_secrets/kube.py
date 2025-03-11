from x_config import XConfigError
from kubernetes import client, config as kube_config

SECRETS_KUBE_SECRET_NAME = 'secrets_kube_secret_name'
SECRETS_KUBE_NAMESPACE = 'secrets_kube_namespace'


def load_kube_secrets(config: dict) -> dict:
    """
    Loads secrets from Kubernetes and returns them in a dict format
    """
    try:
        secret_name = config.pop(SECRETS_KUBE_SECRET_NAME)
    except KeyError:
        raise XConfigError(f'unable to extract `{SECRETS_KUBE_SECRET_NAME} property')

    try:
        namespace = config.pop(SECRETS_KUBE_NAMESPACE)
    except KeyError:
        raise XConfigError(f'unable to extract `{SECRETS_KUBE_NAMESPACE} property')

    kube_config.load_incluster_config()
    v1 = client.CoreV1Api()
    contents = v1.read_namespaced_secret(name=secret_name, namespace=namespace)
    return {k.upper(): v for k, v in contents.items()}
