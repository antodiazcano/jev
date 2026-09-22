"""Configuration of the project."""

from dataclasses import dataclass


@dataclass
class Config:
    """Main configuration class."""

    timeout: int = 10


config = Config()
