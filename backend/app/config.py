"""
Configuration management
Loads configuration uniformly from the .env file in the project root directory
"""

import os
from dotenv import load_dotenv

# Load the .env file from the project root directory
# Path: MiroFish/.env (relative to backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # If no .env exists in the root directory, fall back to loading environment variables (for production)
    load_dotenv(override=True)


class Config:
    """Flask configuration class"""

    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'

    # JSON settings - disable ASCII escaping so non-ASCII characters are rendered directly (instead of \uXXXX format)
    JSON_AS_ASCII = False

    # LLM settings (using OpenAI-compatible format)
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')

    # Zep settings (backward-compatible; can be set to any value in local_zep mode)
    ZEP_API_KEY = os.environ.get('ZEP_API_KEY', 'local')

    # Entity extraction LLM settings (used by local_zep; defaults to the main LLM)
    LLM_EXTRACT_API_KEY = os.environ.get('LLM_EXTRACT_API_KEY') or os.environ.get('LLM_API_KEY')
    LLM_EXTRACT_BASE_URL = os.environ.get('LLM_EXTRACT_BASE_URL') or os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
    LLM_EXTRACT_MODEL_NAME = os.environ.get('LLM_EXTRACT_MODEL_NAME') or os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')
    LLM_EXTRACT_WORKERS = int(os.environ.get('LLM_EXTRACT_WORKERS', '2'))

    # Embedding model for local_zep semantic search (sentence-transformers)
    EMBED_MODEL_NAME = os.environ.get('EMBED_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')

    # local_zep SQLite database path
    LOCAL_ZEP_DB_PATH = os.environ.get(
        'LOCAL_ZEP_DB_PATH',
        os.path.join(os.path.dirname(__file__), '../data/local_zep.db')
    )
    
    # File upload settings
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}

    # Text processing settings
    DEFAULT_CHUNK_SIZE = 500  # default chunk size
    DEFAULT_CHUNK_OVERLAP = 50  # default overlap size

    # OASIS simulation settings
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')

    # OASIS platform available actions
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]
    
    # Report Agent settings
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY is not configured")
        # ZEP_API_KEY is no longer required (not needed in local_zep mode)
        return errors

