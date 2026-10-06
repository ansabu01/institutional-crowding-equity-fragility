"""Create authenticated WRDS connections."""

import logging

import wrds

from ...utils.environment import (
    get_wrds_password,
    get_wrds_username,
)


_LOGGER = logging.getLogger(__name__)


def open_wrds_connection() -> wrds.Connection:
    """Open an authenticated WRDS database connection."""
    _LOGGER.info("Connecting to WRDS.")

    return wrds.Connection(
        wrds_username=get_wrds_username(),
        wrds_password=get_wrds_password(),
    )


__all__ = [
    "open_wrds_connection",
]
