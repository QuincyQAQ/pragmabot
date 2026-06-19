"""OmegaConf-based configuration loader for PragmaBot."""

from omegaconf import OmegaConf, DictConfig
import logging

from pragmabot.utils import get_package_path

# Set up logging
logger = logging.getLogger(__name__)


def get_config() -> DictConfig:
    """Load and return the configuration from the default YAML file.

    Returns:
        The loaded OmegaConf configuration object.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = get_package_path() / "config" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    # Load configuration using OmegaConf
    cfg = OmegaConf.load(config_path)
    logger.info("Configuration loaded from: %s", config_path)
    return cfg
