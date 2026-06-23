"""
全局日志配置
统一管理项目的日志输出，支持控制台和文件双输出
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
import sys


def get_logger(name: str = "paperpilot") -> logging.Logger:
    """
    获取全局日志器
    Args:
        name: 日志器名称，默认 "paperpilot"
    Returns:
        配置好的 Logger 对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 日志格式：[时间] [级别] [模块] 消息
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(module)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器 - 只输出 INFO 及以上级别
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 文件处理器 - 输出 DEBUG 及以上级别，带滚动
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"paperpilot_{datetime.now().strftime('%Y%m%d')}.log"

    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=10,  # 保留10个备份
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# 创建全局日志器实例
logger = get_logger()