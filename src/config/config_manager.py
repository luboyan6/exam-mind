"""配置管理器 —— 加载 YAML 设置和 XML 提示词模板。

提供缓存的、线程安全的接口，供应用全局访问系统参数和提示词字符串。
"""

from __future__ import annotations

import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
_SETTINGS_PATH = _CONFIG_DIR / "settings.yaml"
_PROMPTS_DIR = _CONFIG_DIR / "prompts"

_cache_lock = threading.Lock()
_settings_cache: dict | None = None
_prompt_cache: dict[str, str] = {}


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_xml_prompt(path: Path) -> str:
    tree = ET.parse(path)
    root = tree.getroot()
    text = "".join(root.itertext())
    return text.strip()


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def load_settings(*, reload: bool = False) -> dict:
    """加载并缓存 settings.yaml 中的设置。

    文件不存在时返回空字典（优雅降级）。
    """
    global _settings_cache
    with _cache_lock:
        if _settings_cache is None or reload:
            try:
                _settings_cache = _load_yaml(_SETTINGS_PATH)
            except FileNotFoundError:
                _settings_cache = {}
        return _settings_cache


def get_setting(key: str, default: Any = None) -> Any:
    """通过点号路径访问设置项（如 ``academic.max_retries``）。

    键路径不存在时返回 *default*。
    """
    settings = load_settings()
    current: Any = settings
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def load_prompt(name: str, *, reload: bool = False) -> str:
    """按名称加载并缓存 XML 提示词模板。

    查找 ``config/prompts/{name}.xml``。文件不存在时抛出 ``FileNotFoundError``。
    """
    with _cache_lock:
        if name not in _prompt_cache or reload:
            path = _PROMPTS_DIR / f"{name}.xml"
            if not path.exists():
                raise FileNotFoundError(f"Prompt file not found: {path}")
            _prompt_cache[name] = _load_xml_prompt(path)
        return _prompt_cache[name]


def clear_cache() -> None:
    """使所有缓存的设置和提示词失效。"""
    global _settings_cache
    with _cache_lock:
        _settings_cache = None
        _prompt_cache.clear()

