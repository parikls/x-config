import json
import logging
import os
from enum import Enum
from pathlib import Path
from pydoc import locate
from typing import Type

import boto3
import yaml
from dotenv import load_dotenv
from mako.lookup import TemplateLookup
from pydantic import create_model, BaseModel

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent


class ConfigurationError(Exception):
    """
    Base configuration error
    """


class X:

    @classmethod
    def config(
            cls,
            root_dir: Path,
            aws_region: str = 'us-east-1'
    ):

        try:
            env = os.environ['ENV']
        except KeyError:
            raise ConfigurationError('X.config: environment variable `ENV` must be set')

        config_path = root_dir.parent / 'config.yaml'
        if not config_path.exists():
            raise ConfigurationError(f'X.config: config.yaml was not found at {config_path}')

        with config_path.open() as f:
            full_config = yaml.load(f, Loader=yaml.CLoader)

        try:
            def_constants = full_config.pop('constants')
        except KeyError:
            raise ConfigurationError('X.config: section `constants` does not exist in a config.yaml file')

        try:
            def_secrets = full_config.pop('secrets')
        except KeyError:
            raise ConfigurationError('X.config: section `secrets` does not exist in a config.yaml file')

        floating_config = cls._validate_and_get_floating_config(env=env, full_config=full_config)

        Env = Enum('Env', {x.upper(): x for x in [env, *full_config.keys()]})
        env = Env(env)

        constants_model = cls._create_constants_model(def_constants)
        secrets_model = cls._create_secrets_model(def_secrets)
        floating_model = cls._create_floating_model(floating_config)

        if floating_config['dotenv_secrets']:
            secrets = cls.secrets_from_dotenv(root_dir, dotenv_name=floating_config['dotenv_name'])
        else:
            secrets = cls.secrets_from_aws(env, floating_config['app_name'], aws_region)

        secrets = secrets_model.model_validate(secrets)

        constants, floating = constants_model(), floating_model()

        cls._render_pyi(root_dir, constants, floating_model, secrets_model, envs=Env)

        return cls(
            constants=constants,
            floating=floating,
            secrets=secrets,
            env=env,
            Env=Env
        )

    @classmethod
    def _create_secrets_model(cls, def_secrets):
        return create_model(
            'Secrets',
            **{k.upper(): (locate(v), ...) for k, v in def_secrets.items()}
        )

    @classmethod
    def _create_constants_model(cls, def_constants):
        return create_model(
            'Constants',
            **{k.upper(): (type(v), v) for k, v in def_constants.items()}
        )

    @classmethod
    def _create_floating_model(cls, def_floating):
        return create_model(
            'Floating',
            **{k.upper(): (type(v), v) for k, v in def_floating.items()}
        )

    @classmethod
    def _validate_and_get_floating_config(cls, env: str, full_config: dict):
        try:
            base_section = full_config.pop('base')
        except KeyError:
            raise ConfigurationError(f'{cls}: section `base` does not exist in a config.yaml file')

        try:
            env_section = full_config.pop(env)
        except KeyError:
            raise ConfigurationError(f'{cls}: section `{env}` does not exist in a config.yaml file, '
                                     f'though `ENV` environment variable is set to a `{env}`')

        env_config = {**base_section, **env_section}

        # now let's ensure that this env section has all the keys that all other envs has, and vice versa
        for other_env, other_env_section in full_config.items():
            other_env_config = {**base_section, **other_env_section}
            for key, value in other_env_config.items():
                value_type = type(value)

                if key not in env_config:
                    raise ConfigurationError(f'{cls}: key `{key}` is missing in env `{env}`, '
                                             f'but present in env `{other_env}`')

                env_type = type(env_config[key])
                if value_type is not env_type:
                    raise ConfigurationError(f'{cls}: key `{env}.{key}` is of type {env_type}, '
                                             f'while {other_env}.{key} is of type {value_type}')

            for env_key, env_value in env_config.items():
                if env_key not in other_env_config:
                    raise ConfigurationError(f'{cls}: key `{env_key}` is missing in env `{other_env}`, '
                                             f'but present in env `{env}`')

        return env_config

    @classmethod
    def secrets_from_dotenv(
            cls,
            root_dir: Path,
            dotenv_name: str,
    ):
        load_dotenv(root_dir.parent / dotenv_name)
        return {k.upper(): v for k, v in os.environ.items()}

    @classmethod
    def secrets_from_aws(cls, env, app_name: str, region: str):
        secret_name = f'{env}/{app_name}'

        # Create a Secrets Manager client
        session = boto3.session.Session()
        client = session.client(service_name='secretsmanager', region_name=region)
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        return json.loads(get_secret_value_response['SecretString'])

    @classmethod
    def _render_pyi(
            cls,
            root_dir: Path,
            constants: BaseModel,
            floating_model: Type[BaseModel],
            secrets_model: Type[BaseModel],
            envs: Type[Enum]
    ):
        constants_def = []
        for key, type_ in constants.__annotations__.items():
            if type_ is str:
                value = f"'{getattr(constants, key)}'"
            else:
                value = getattr(constants, key)
            constants_def.append((key, type_.__name__, value))

        lookup = TemplateLookup(directories=[HERE], filesystem_checks=False)
        template = lookup.get_template("template.mako")
        rendered = template.render(
            constants=constants_def,
            floating=floating_model,
            secrets=secrets_model,
            envs=envs
        )
        with (root_dir / '__init__.pyi').open('w') as f:
            f.write(rendered)

    def __init__(
            self,
            constants: BaseModel,
            floating: BaseModel,
            secrets: BaseModel,
            env: Enum,
            Env: Type[Enum]  # noqa
    ):
        self.constants = constants
        self.floating = floating
        self.secrets = secrets
        self.env = env
        self.Env = Env
