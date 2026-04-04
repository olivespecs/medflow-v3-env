"""Validator-friendly app export for OpenEnv tooling."""

from src.main import app
from src.main import main as _main

__all__ = ["app", "main"]


def main() -> None:
    """Console script entrypoint used by OpenEnv validators."""
    _main()


if __name__ == "__main__":
    main()
