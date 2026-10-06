"""Load and validate environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _PROJECT_ROOT / ".env"

load_dotenv(_ENV_FILE)


@lru_cache
def _get_required_env(name: str) -> str:
    """Return a required environment variable or raise an error."""
    value = os.getenv(name)

    if not value or not value.strip():
        raise RuntimeError(f"Required environment variable {name!r} is missing.")

    return value.strip()


@lru_cache
def get_sec_contact_email() -> str:
    """Return and validate the contact email used for SEC requests."""
    email = _get_required_env("SEC_CONTACT_EMAIL")

    if (
        "@" not in email
        or email.startswith("@")
        or email.endswith("@")
        or any(character.isspace() for character in email)
        or "\r" in email
        or "\n" in email
    ):
        raise RuntimeError("SEC_CONTACT_EMAIL is not a valid contact email.")

    return email


@lru_cache
def get_wrds_username() -> str:
    """Return the username used to authenticate with WRDS."""
    username = _get_required_env("WRDS_USERNAME")

    if any(character.isspace() for character in username):
        raise RuntimeError("WRDS_USERNAME must not contain whitespace.")

    return username


@lru_cache
def get_wrds_password() -> str:
    """Return the password used to authenticate with WRDS."""
    password = os.getenv("WRDS_PASSWORD")

    if password is None or not password:
        raise RuntimeError("Required environment variable 'WRDS_PASSWORD' is missing.")

    if "\r" in password or "\n" in password:
        raise RuntimeError("WRDS_PASSWORD must not contain newline characters.")

    return password


__all__ = [
    "get_sec_contact_email",
    "get_wrds_password",
    "get_wrds_username",
]
