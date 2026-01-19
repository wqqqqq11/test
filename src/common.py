import json
import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict


def load_config(path: str = "config.json") -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def setup_logger(name: str, config: Dict[str, Any]) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(config['logging']['level'])
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(config['logging']['format']))
        logger.addHandler(handler)
    
    return logger


def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
