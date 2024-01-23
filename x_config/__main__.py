from argparse import ArgumentParser

from x_config import X


def main():
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
    floating_model = X.create_floating_model(config=full_config)
    Env = X.create_env_enum(full_config)  # noqa
    X.render_pyi(app_dir, constants_model, floating_model, secrets_model, envs=Env)


if __name__ == '__main__':
    main()