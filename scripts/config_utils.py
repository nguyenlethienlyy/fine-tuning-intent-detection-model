"""Shared helpers for loading and validating configuration files."""

import os
import yaml


class ConfigError(Exception):
    """Raised when a configuration file is missing, unreadable or incomplete."""


def load_config(config_path):
    """Load a YAML config file, failing loudly with an actionable message."""
    if not os.path.isfile(config_path):
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in config file {config_path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigError(f"Config file {config_path} must contain a YAML mapping")

    return config


def require(config, path, config_path):
    """Return a nested config value, raising ConfigError when it is missing."""
    value = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            missing = ".".join(path)
            raise ConfigError(f"Missing required key '{missing}' in config file {config_path}")
        value = value[key]
    return value


def require_file(file_path, description):
    """Raise a clear error when an input file the pipeline depends on is absent."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{description} not found: {file_path}")
    return file_path
