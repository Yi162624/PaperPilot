"""
论文数据存储
管理论文元数据的持久化存储
"""

import json
import hashlib
from pathlib import Path
from typing import Optional, Dict

from logger import logger

# 存储目录和文件
_DATA_DIR = Path("./data")
_DATA_FILE = _DATA_DIR / "papers_data.json"

# 内存存储
_papers_storage: Dict = {}


def _load_data():
    """从文件加载论文数据"""
    if _DATA_FILE.exists():
        try:
            with open(_DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            _papers_storage.update(data.get("papers", {}))
            logger.info(f"已加载 {len(_papers_storage)} 篇论文数据")
        except Exception as e:
            logger.warning(f"加载持久化数据失败: {e}")


def _save_data():
    """持久化论文数据到文件"""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump({"papers": _papers_storage}, f, ensure_ascii=False, indent=2, default=str)


def get_papers_storage() -> Dict:
    """获取论文存储字典"""
    return _papers_storage


def get_paper(paper_id: str) -> Optional[Dict]:
    """获取单篇论文数据"""
    return _papers_storage.get(paper_id)


def add_paper(paper_id: str, paper_data: Dict):
    """添加论文到存储"""
    _papers_storage[paper_id] = paper_data
    _save_data()
    logger.info(f"已添加论文: {paper_data.get('title', paper_id)}")


def update_paper(paper_id: str, paper_data: Dict):
    """更新论文数据"""
    if paper_id in _papers_storage:
        _papers_storage[paper_id].update(paper_data)
        _save_data()


def delete_paper(paper_id: str):
    """删除论文"""
    if paper_id in _papers_storage:
        del _papers_storage[paper_id]
        _save_data()
        logger.info(f"已删除论文: {paper_id}")


def calculate_file_hash(content: bytes) -> str:
    """计算文件 SHA256 哈希"""
    return hashlib.sha256(content).hexdigest()


def get_paper_by_hash(file_hash: str) -> tuple[Optional[str], Optional[Dict]]:
    """根据文件哈希查找已有论文"""
    for pid, p in _papers_storage.items():
        if p.get("file_hash") == file_hash:
            return pid, p
    return None, None


def init_storage():
    """初始化存储（启动时调用）"""
    _load_data()


def save_storage():
    """保存存储状态"""
    _save_data()


# 启动时自动加载
init_storage()