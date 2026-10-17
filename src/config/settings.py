import yaml
from pathlib import Path


def load_config() -> dict:
   """Load settings from config.yaml next to this file."""
   # utf-8 explicitly: the file has non-ASCII comments, and Windows would default to cp1252
   return yaml.safe_load((Path(__file__).parent / "config.yaml").read_text(encoding="utf-8"))


config = load_config()
