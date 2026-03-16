"""
local_zep — zep_cloud 本地替代实现
无需 Zep Cloud 账号，使用 SQLite + 本地 LLM 提供知识图谱功能
"""

import os

from ..config import Config
from . import db as _db
from .client import LocalZep as Zep
from .models import EpisodeData, EntityEdgeSourceTarget, InternalServerError

# 初始化数据库（模块导入时执行一次）
_db.init(Config.LOCAL_ZEP_DB_PATH)

__all__ = ["Zep", "EpisodeData", "EntityEdgeSourceTarget", "InternalServerError"]
