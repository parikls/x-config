from argparse import ArgumentParser

from x_config import X


def main():
    """
    Console script to generate an initial `__init__.pyi` file
    """
    parser = ArgumentParser()
    parser.add_argument(
        '--config-path',
        '-c',
        required=False,
        default=None,
        help='Absolute path to a config yaml file (including filename)'
    )
    parser.add_argument(
        '--app-dir',
        '-a',
        required=False,
        default=None,
        help='Absolute path to a root python module of your app'
    )
    args = parser.parse_args()
    config_dir, _, app_dir = X.ensure_paths(
        config_path=args.config_path,
        app_dir=args.app_dir
    )
    full_config = X.load_full_config(config_dir)
    constants_model = X.create_constants_model(config=full_config)
    secrets_model = X.create_secrets_model(config=full_config)
    env_model = X.create_env_model(config=full_config)
    env_choices = X.create_env_choices_enum(full_config)
    X.render_pyi(app_dir, constants_model, env_model, secrets_model, env_choices=env_choices)


if __name__ == '__main__':
    main()
