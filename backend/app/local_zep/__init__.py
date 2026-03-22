"""
local_zep — local drop-in replacement for zep_cloud
No Zep Cloud account required; uses SQLite + a local LLM to provide knowledge graph functionality.
"""

import os

from ..config import Config
from . import db as _db
from .client import LocalZep as Zep
from .models import EpisodeData, EntityEdgeSourceTarget, InternalServerError

# Initialize the database (executed once at module import time)
_db.init(Config.LOCAL_ZEP_DB_PATH)

__all__ = ["Zep", "EpisodeData", "EntityEdgeSourceTarget", "InternalServerError"]
