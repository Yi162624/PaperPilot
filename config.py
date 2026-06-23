"""
配置管理
加载环境变量和项目配置
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """项目配置类"""

    # Kimi API 配置
    kimi_api_key: str = ""
    kimi_base_url: str = "https://api.moonshot.cn/v1"
    kimi_model: str = "kimi-k2.6"

    # DeepSeek API 配置（用于 ReAct Agent 决策）
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-pro"

    # 存储配置
    upload_dir: str = "./uploads"
    image_output_dir: str = "./images"
    data_dir: str = "./data"
    max_file_size: int = 50 * 1024 * 1024  # 50MB

    class Config:
        env_file = str(Path(__file__).parent / ".env")
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()