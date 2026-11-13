import logging
import sys


def setup_logger(
    name: str = "ensemble_learning",
    level: int = logging.INFO,
    format_str: str = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """Configure and return a standardized logger instance.

    Parameters
    ----------
    name : str, optional
        Logger name, by default "ensemble_learning".
    level : int, optional
        Logging level, by default logging.INFO.
    format_str : str, optional
        Log message format string.
    datefmt : str, optional
        Date format string for timestamps.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if logger was already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(fmt=format_str, datefmt=datefmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger
