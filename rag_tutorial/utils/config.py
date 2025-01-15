import yaml

def read_config(config_path: str) -> dict:
    """
    Read the configuration from a YAML file.

    :param config_path: Path to the YAML configuration file.
    :return: A dictionary containing the configuration.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
