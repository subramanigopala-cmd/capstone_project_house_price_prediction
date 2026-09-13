import os
import runpy
from pathlib import Path

# path to common config.py
config_path = Path(__file__).resolve().parents[1] / "config.py"

# run it in an isolated namespace
config_globals = runpy.run_path(str(config_path))
VERSION = config_globals["VERSION"]
