class XConfigError(Exception):
    """
    Base configuration error
    """

    def __init__(self, msg):
        super().__init__(f'X.Config: {msg}')

