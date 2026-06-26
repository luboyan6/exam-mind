"""配置模块 —— YAML 设置 + XML 提示词注册表。

公开 API:
    load_settings()   — 加载并缓存 settings.yaml
    get_setting(key)  — 通过点号路径访问设置
    load_prompt(name) — 加载并缓存 XML 提示词模板
    clear_cache()     — 使所有缓存失效
"""

from src.config.config_manager import clear_cache, get_setting, load_prompt, load_settings

__all__ = [
    "clear_cache",
    "get_setting",
    "load_prompt",
    "load_settings",
]

