import logging
import os
from enum import Enum
from pathlib import Path
from pydoc import locate
from typing import Type, Callable

import yaml
from mako.lookup import TemplateLookup
from pydantic import create_model, BaseModel
from yaml import YAMLError

from x_config.x_secrets import SecretsSource, SECRET_SOURCE_REQUIRED_PROPS, SECRET_SPECIFIC_PROPS
from x_config.x_secrets.aws import load_aws_secrets
from x_config.x_secrets.dotenv import load_dotenv_secrets

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent

CONFIG_SECTION_CONSTANTS = 'constants'
CONFIG_SECTION_SECRETS = 'secrets'
CONFIG_SECTION_BASE = 'base'

SECRETS_SOURCE_PROP_NAME = 'secrets_source'


class ConfigurationError(Exception):
    """
    Base configuration error
    """


class X:

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

    @classmethod
    def config(
            cls,
            *,
            config_path: str | Path = None,
            dotenv_dir: str | Path = None,
            app_dir: str | Path = None,
            aws_region: str = 'us-east-1'
    ):
        """
        entry-point for configuration

        :param config_path: a path to a dir where `config.yaml` sits
        :param dotenv_dir: a path to a dir where dotenv file sits
        :param app_dir: a path to a root of app directory (e.g., your root python module)
        :param aws_region: AWS Secrets region
        """
        try:
            env = os.environ['ENV']
        except KeyError:
            raise ConfigurationError('X.config: environment variable `ENV` must be set')

        config_path, dotenv_dir, app_dir = cls.ensure_paths(config_path, dotenv_dir, app_dir)

        full_config = cls.load_full_config(config_path)
        constants_model = cls.create_constants_model(config=full_config)
        secrets_model = cls.create_secrets_model(config=full_config)
        floating_model = cls.create_floating_model(config=full_config)
        Env = cls.create_env_enum(full_config)  # noqa
        cls.render_pyi(app_dir, constants_model, floating_model, secrets_model, envs=Env)

        floating_config = cls._validate_and_get_floating_config(full_config=full_config, env=env)
        secrets_source = cls._get_secrets_source(floating_config=floating_config)

        # populate constants with defaults
        constants = constants_model()

        # populate floating from config
        floating = floating_model.model_validate(floating_config)

        # populate secrets either from .env or from AWS secrets
        if secrets_source is SecretsSource.AWS:
            secrets = load_aws_secrets(secret_name=floating.SECRETS_AWS_SECRET_NAME, region=aws_region)
        else:
            secrets = load_dotenv_secrets(dotenv_dir=dotenv_dir, dotenv_name=floating.SECRETS_DOTENV_NAME)
        secrets = secrets_model.model_validate(secrets)

        return cls(
            constants=constants,
            floating=floating,
            secrets=secrets,
            env=Env(env),
            Env=Env
        )

    @classmethod
    def ensure_paths(
            cls,
            config_path: str | Path = None,
            dotenv_dir: str | Path = None,
            app_dir: str | Path = None
    ) -> tuple[Path, Path, Path]:
        """
        :param config_path:
        :param dotenv_dir:
        :param app_dir:
        :return:
        """

        # will use default dirs based on a current dir.
        # can be useful for cli command to generate `.pyi` file
        default_app_dir = Path(os.getcwd())
        default_root_dir = default_app_dir.parent

        # ensure that directories are exists
        try:
            config_path = (
                Path(config_path) if config_path else Path(default_root_dir) / 'config.yaml'
            )
        except TypeError:
            raise ConfigurationError(f'X.config: invalid config_path {config_path}')

        try:
            dotenv_dir = Path(dotenv_dir or default_root_dir)
        except TypeError:
            raise ConfigurationError(f'X.config: invalid dotenv_dir {dotenv_dir}')

        try:
            app_dir = Path(app_dir or default_app_dir)
        except TypeError:
            raise ConfigurationError(f'X.config: invalid app_dir {app_dir}')

        for path in (config_path, dotenv_dir, app_dir):
            if not path.exists():
                raise ConfigurationError(f'X.config: {path} does not exists')

        return config_path, dotenv_dir, app_dir

    @classmethod
    def load_full_config(cls, config_path: Path):
        """
        Loads all the data from the config.yaml file
        """
        with config_path.open() as f:
            try:
                return yaml.load(f, Loader=yaml.CSafeLoader)
            except YAMLError:
                raise ConfigurationError(f'X.config: invalid yaml file: {config_path}')

    @classmethod
    def create_constants_model(cls, config: dict):
        try:
            constants = config.pop(CONFIG_SECTION_CONSTANTS)
        except KeyError:
            raise ConfigurationError(f'X.config: section `{CONFIG_SECTION_CONSTANTS}` '
                                     f'does not exist in a config.yaml file')

        return cls._create_pydantic_model(
            'Constants',
            constants,
            type_func=type,
            use_value_as_default=True
        )

    @classmethod
    def create_floating_model(cls, config: dict):
        # pick any non-base section
        any_section = [x for x in config.keys() if x != CONFIG_SECTION_BASE][0]

        # merge with base
        definition = {**{k: v for k, v in config[CONFIG_SECTION_BASE].items()}, **{k: v for k, v in config[any_section].items()}}

        return cls._create_pydantic_model(
            'Floating',
            config_contents=definition,
            type_func=type,
            use_value_as_default=False
        )

    @classmethod
    def create_secrets_model(cls, config: dict):
        try:
            secrets = config.pop('secrets')
        except KeyError:
            raise ConfigurationError(
                f'X.config: section `{CONFIG_SECTION_SECRETS}` does not exist in a config.yaml file'
            )
        return cls._create_pydantic_model(
            'Secrets',
            secrets,
            type_func=locate,
            use_value_as_default=False
        )

    @classmethod
    def create_env_enum(cls, config: dict):
        return Enum(
            'Env',
            {x.upper(): x for x in [x for x in config.keys() if x != CONFIG_SECTION_BASE]}
        )


    @classmethod
    def render_pyi(
            cls,
            app_dir: Path,
            constants_model: Type[BaseModel],
            floating_model: Type[BaseModel],
            secrets_model: Type[BaseModel],
            envs: Type[Enum]
    ):
        """
        Renders __init__.pyi file which will be used by an IDE for autocompletion
        """
        constants_def = []
        constants = constants_model()
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
        with (app_dir / '__init__.pyi').open('w') as f:
            f.write(rendered)


    @classmethod
    def _create_pydantic_model(
            cls,
            model_name: str,
            config_contents: dict,
            type_func: Callable,
            use_value_as_default: bool
    ):
        """
        Creates a pydantic model based on a config section provided (key-values from yaml).

        :param model_name: future pydantic model name
        :param config_contents: config contents
        :param type_func: func to determine a type of a future variable
        :param use_value_as_default: whether to populate a default value for that model

        Example:

        >>> cls._create_pydantic_model(
        >>>     model_name='MyModel',
        >>>     config_contents={'a': 1, 'b': 'b'},
        >>>     type_func=type,
        >>>     use_value_as_default=True
        >>> )

        will produce

        >>> from pydantic import BaseModel
        >>>
        >>> class MyModel(BaseModel):
        >>>     a: int = 1
        >>>     b: str = 'b'

        """
        return create_model(
            model_name,
            **{
                k.upper(): (type_func(v), v if use_value_as_default else ...)
                for k, v in config_contents.items()
            }
        )

    @classmethod
    def _validate_and_get_floating_config(cls, full_config: dict, env: str):
        """
        Merges base + selected-env config and returns.
        Also performs a validation by comparing a current selected section
        to all other sections, which includes:
        - checks the presence of the keys
        - compare data types
        """
        try:
            base_section = full_config.pop(CONFIG_SECTION_BASE)
        except KeyError:
            raise ConfigurationError(
                f'X.Config: section `{CONFIG_SECTION_BASE}` does not exist in a config.yaml file')

        try:
            env_section = full_config.pop(env)
        except KeyError:
            raise ConfigurationError(f'.Config: section `{env}` does not exist in a config.yaml file, '
                                     f'though `ENV` environment variable is set to a `{env}`')

        # merge env-specific and base sections
        floating_config = {**base_section, **env_section}
        env_keys = {k for k in floating_config.keys() if k not in SECRET_SPECIFIC_PROPS}

        # ensure that this env section has all the keys that all other envs have, and vice versa
        for other_env, other_env_section in full_config.items():
            other_env_config = {**base_section, **other_env_section}
            other_keys = {k for k in other_env_config.keys() if k not in SECRET_SPECIFIC_PROPS}
            missing_keys = other_keys - env_keys
            if missing_keys:
                raise ConfigurationError(f'X.Config: keys `{missing_keys}` are missing in '
                                         f'env `{env}`, but present in env `{other_env}`')

            extra_keys = env_keys - other_keys
            if extra_keys:
                raise ConfigurationError(f'.Config: keys `{extra_keys}` are missing in '
                                         f'env `{other_env}`, but present in env `{env}`')

            # now let's compare types
            for key in env_keys:
                env_type = type(floating_config[key])
                other_env_type = type(other_env_config[key])
                if env_type is not other_env_type:
                    raise ConfigurationError(
                        f'X.Config: key `{env}.{key}` is of type {env_type}, '
                        f'while {other_env}.{key} is of type {other_env_type}'
                    )

        return {k.upper(): v for k, v in floating_config.items()}

    @classmethod
    def _get_secrets_source(cls, floating_config: dict) -> SecretsSource:
        try:
            secrets_source = floating_config[SECRETS_SOURCE_PROP_NAME.upper()]
        except KeyError:
            raise ConfigurationError(
                f'X.config: `{SECRETS_SOURCE_PROP_NAME}` property was not found in config'
            )

        try:
            secrets_source = SecretsSource(secrets_source)
        except ValueError:
            raise ConfigurationError(f'X.config: unknown source of secrets {secrets_source}')

        for required_secrets_prop in SECRET_SOURCE_REQUIRED_PROPS[secrets_source]:
            if required_secrets_prop.upper() not in floating_config:
                raise ConfigurationError(
                    f'X.config: `{required_secrets_prop}` property is a required '
                    f'for `{secrets_source}` secrets source'
                )

        return secrets_source
