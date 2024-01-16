"""
This is a helper file created by X-Config specifically for annotations
"""


from enum import Enum
from typing import Type
from pydantic import BaseModel


class Env(Enum):
% for env in envs:
    ${env.name.upper()} = '${env.value}'
% endfor


class Constants(BaseModel):
% for key, type, value in constants:
    ${key}: ${type} = ${value}
% endfor


class Floating(BaseModel):
% if floating:
% for key, type in floating.__annotations__.items():
    ${key}: ${type.__name__}
% endfor
% else:
    pass
% endif


class Secrets(BaseModel):
% if secrets:
% for key, type in secrets.__annotations__.items():
    ${key}: ${type.__name__}
% endfor
% else:
    pass
% endif


class Config:
    constants: Constants
    floating: Floating
    secrets: Secrets
    env: Env
    Env: Type[Env]


config: Config