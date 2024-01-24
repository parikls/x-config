import logging
import os
from enum import Enum
from pathlib import Path
from pydoc import locate
from typing import Type, Callable

import yaml
from mako.exceptions import MakoException
from mako.lookup import TemplateLookup
from pydantic import create_model, BaseModel
from yaml import YAMLError

from x_config.errors import XConfigError
from x_config.x_secrets import (
    SecretsSource, SECRET_SOURCE_REQUIRED_PROPS, SECRET_SPECIFIC_PROPS,
    SECRETS_SOURCE_PROP_NAME
)
from x_config.x_secrets.aws import load_aws_secrets
from x_config.x_secrets.dotenv import load_dotenv_secrets

logger = logging.getLogger(__name__)

CONFIG_SECTION_CONSTANTS = 'constants'
CONFIG_SECTION_SECRETS = 'secrets'
CONFIG_SECTION_BASE = 'base'


class X:

    def __init__(
            self,
            x_const: BaseModel,
            x_env: BaseModel,
            x_secret: BaseModel,
            current_environment: Enum,
            environment_choices: Type[Enum]
    ):
        self.x_const = x_const
        self.x_env = x_env
        self.x_secret = x_secret
        self.current_environment = current_environment
        self.environment_choices = environment_choices

    @classmethod
    def config(
            cls,
            *,
            config_path: str | Path = None,
            dotenv_dir: str | Path = None,
            app_dir: str | Path = None
    ):
        """
        entry-point for configuration

        :param config_path: a path to a dir where `config.yaml` sits
        :param dotenv_dir: a path to a dir where dotenv file sits
        :param app_dir: a path to a root of app directory (e.g., your root python module)
        """
        try:
            env = os.environ['ENV']
        except KeyError:
            raise XConfigError('environment variable `ENV` must be set')

        config_path, dotenv_dir, app_dir = cls.ensure_paths(config_path, dotenv_dir, app_dir)

        full_config = cls.load_full_config(config_path)
        constants_model = cls.create_constants_model(config=full_config)
        secrets_model = cls.create_secrets_model(config=full_config)
        env_model = cls.create_env_model(config=full_config)
        env_choices = cls.create_env_choices_enum(full_config)
        cls.render_pyi(
            app_dir=app_dir,
            constants_model=constants_model,
            env_model=env_model,
            secrets_model=secrets_model,
            env_choices=env_choices
        )

        env_config = cls._validate_and_get_env_config(full_config=full_config, env=env)

        # populate constants with defaults
        const_vars = constants_model()

        # populate secrets
        secrets_vars = cls._populate_secrets(
            secrets_model=secrets_model,
            config=env_config,
            dotenv_dir=dotenv_dir
        )

        # populate env config
        env_vars = env_model.model_validate({k.upper(): v for k, v in env_config.items()})

        return cls(
            x_const=const_vars,
            x_env=env_vars,
            x_secret=secrets_vars,
            current_environment=env_choices(env),
            environment_choices=env_choices
        )

    @classmethod
    def ensure_paths(
            cls,
            config_path: str | Path = None,
            dotenv_dir: str | Path = None,
            app_dir: str | Path = None
    ) -> tuple[Path, Path, Path]:
        """
        Ensures that all provided paths exist
        """

        # will use default dirs based on a current dir.
        # can be useful for cli command to generate an `__init__.pyi` file,
        # so user won't need to pass extra cli args
        default_app_dir = Path(os.getcwd())
        default_root_dir = default_app_dir.parent

        # ensure that directories are exists
        try:
            config_path = (
                Path(config_path) if config_path else Path(default_root_dir) / 'config.yaml'
            )
        except TypeError:
            raise XConfigError(f'invalid config_path {config_path}')

        try:
            dotenv_dir = Path(dotenv_dir or default_root_dir)
        except TypeError:
            raise XConfigError(f'invalid dotenv_dir {dotenv_dir}')

        try:
            app_dir = Path(app_dir or default_app_dir)
        except TypeError:
            raise XConfigError(f'invalid app_dir {app_dir}')

        for path in (config_path, dotenv_dir, app_dir):
            if not path.exists():
                raise XConfigError(f'{path} does not exists')

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
                raise XConfigError(f'invalid yaml file: {config_path}')

    @classmethod
    def create_constants_model(cls, config: dict) -> Type[BaseModel]:
        """
        Creates a pydantic model based on provided `config`.
        Will pop `constants` section out of it
        """
        try:
            constants = config.pop(CONFIG_SECTION_CONSTANTS)
        except KeyError:
            raise XConfigError(
                f'section `{CONFIG_SECTION_CONSTANTS}` does not exist in a config.yaml file'
            )

        return cls._create_pydantic_model(
            'Constants',
            constants,
            type_func=type,
            use_value_as_default=True
        )

    @classmethod
    def create_secrets_model(cls, config: dict) -> Type[BaseModel]:
        """
        Creates a pydantic model based on a provided `config`.
        Will pop `secrets` section out of it
        """
        try:
            secrets = config.pop(CONFIG_SECTION_SECRETS)
        except KeyError:
            raise XConfigError(
                f'section `{CONFIG_SECTION_SECRETS}` does not exist in a config.yaml file'
            )
        return cls._create_pydantic_model(
            'Secrets',
            secrets,
            type_func=locate,
            use_value_as_default=False
        )

    @classmethod
    def create_env_model(cls, config: dict) -> Type[BaseModel]:
        """
        Creates an environment-specific pydantic model based on a provided `config`.
        Will use any environment section to get all the necessary env keys
        """
        # at this point we've already popped out constants and secrets,
        # therefore, we may pick an any non-base section
        any_section = [x for x in config.keys() if x != CONFIG_SECTION_BASE][0]

        # merge `base` section with an env-specific section
        definition = {
            **{k: v for k, v in config[CONFIG_SECTION_BASE].items()},
            **{k: v for k, v in config[any_section].items()}
        }

        # remove secret-specific props
        definition = {k: v for k, v in definition.items() if k not in SECRET_SPECIFIC_PROPS}

        return cls._create_pydantic_model(
            'Env',
            config_contents=definition,
            type_func=type,
            use_value_as_default=False
        )

    @classmethod
    def create_env_choices_enum(cls, config: dict) -> Type[Enum]:
        """
        Returns an enum with all possible environments choices
        """
        return Enum(
            'Environment',
            {x.upper(): x for x in [x for x in config.keys() if x != CONFIG_SECTION_BASE]}
        )

    @classmethod
    def render_pyi(
            cls,
            app_dir: Path,
            constants_model: Type[BaseModel],
            env_model: Type[BaseModel],
            secrets_model: Type[BaseModel],
            env_choices: Type[Enum]
    ):
        """
        Renders __init__.pyi file in the app_dir which will be used by an IDE for autocompletion
        """

        # prepare a constants definition to populate `.pyi` file with defaults
        constants_def = []
        constants = constants_model()
        for key, type_ in constants.__annotations__.items():
            value = getattr(constants, key)
            if type_ is str:
                value = f"'{value}'"  # enclose value in a single parenthesis
            constants_def.append((key, type_.__name__, value))

        here = Path(__file__).parent
        lookup = TemplateLookup(directories=[here], filesystem_checks=False)
        try:
            template = lookup.get_template("template.mako")
        except MakoException:
            raise XConfigError('internal error. unable to find `template.mako` file')

        try:
            rendered = template.render(
                constants=constants_def,
                env_vars_model=env_model,
                secret_vars_model=secrets_model,
                env_choices=env_choices
            )
        except MakoException:
            raise XConfigError('internal error. unable to render template')

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
    def _validate_and_get_env_config(cls, full_config: dict, env: str) -> dict:
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
            raise XConfigError(
                f'section `{CONFIG_SECTION_BASE}` does not exist in a config.yaml file')

        try:
            env_section = full_config.pop(env)
        except KeyError:
            raise XConfigError(f'section `{env}` does not exist in a config.yaml file, '
                               f'though `ENV` environment variable is set to a `{env}`')

        # merge env-specific and base sections
        env_config = {**base_section, **env_section}
        env_keys = {k for k in env_config.keys() if k not in SECRET_SPECIFIC_PROPS}

        # ensure that this env section has all the keys that all other envs have, and vice versa
        for other_env, other_env_section in full_config.items():
            other_env_config = {**base_section, **other_env_section}
            other_keys = {k for k in other_env_config.keys() if k not in SECRET_SPECIFIC_PROPS}
            missing_keys = other_keys - env_keys
            if missing_keys:
                raise XConfigError(f'keys `{missing_keys}` are missing in '
                                   f'env `{env}`, but present in env `{other_env}`')

            extra_keys = env_keys - other_keys
            if extra_keys:
                raise XConfigError(f'keys `{extra_keys}` are missing in '
                                   f'env `{other_env}`, but present in env `{env}`')

            # now let's compare types
            for key in env_keys:
                env_type = type(env_config[key])
                other_env_type = type(other_env_config[key])
                if env_type is not other_env_type:
                    raise XConfigError(
                        f'key `{env}.{key}` is of type {env_type}, '
                        f'while {other_env}.{key} is of type {other_env_type}'
                    )

        return env_config

    @classmethod
    def _populate_secrets(cls,
                          secrets_model: Type[BaseModel],
                          config: dict,
                          dotenv_dir: Path) -> BaseModel:
        """
        Populates `secrets_model` with the secret data
        """

        # load and validate secrets definitions
        try:
            secrets_source = config.pop(SECRETS_SOURCE_PROP_NAME)
        except KeyError:
            raise XConfigError(f'`{SECRETS_SOURCE_PROP_NAME}` property was not found in config')

        try:
            secrets_source = SecretsSource(secrets_source)
        except ValueError:
            raise XConfigError(f'unknown source of secrets {secrets_source}')

        for required_secrets_prop in SECRET_SOURCE_REQUIRED_PROPS[secrets_source]:
            if required_secrets_prop not in config:
                raise XConfigError(
                    f'`{required_secrets_prop}` property is a required '
                    f'for `{secrets_source}` secrets source'
                )

        # do an actual load of secrets
        if secrets_source is SecretsSource.AWS:
            secrets = load_aws_secrets(config=config)
        else:
            secrets = load_dotenv_secrets(dotenv_dir=dotenv_dir, config=config)

        # build and return populated model
        return secrets_model.model_validate(secrets)
