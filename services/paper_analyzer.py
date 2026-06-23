"""
论文分析服务
调用 AI 生成结构化报告、回答用户问题
"""

import re
import time
from typing import Optional, List, Tuple

from clients import KimiClient
from prompts import SYSTEM_PROMPT, QA_PROMPT
from logger import logger


class PaperAnalyzer:
    """论文分析器：调用 Kimi 生成结构化报告、回答用户问题"""

    def __init__(self, kimi_client: Optional[KimiClient] = None):
        self.kimi = kimi_client or KimiClient()  # 初始化 Kimi 远端大模型客户端


    def analyze_paper(self, paper_id: str, paper_text: str, images: Optional[List[str]] = None) -> str:
        """分析论文并生成结构化报告（支持传入图片做多模态分析）"""
        logger.info(f"开始分析论文: {paper_id}, 图片数: {len(images) if images else 0}")
        paper_text = self._strip_references(paper_text)  # 去掉参考文献条目，省 token
        full_prompt = f"{SYSTEM_PROMPT}\n\n请分析以下论文并生成结构化报告：\n\n{paper_text[:190000]}"
        time.sleep(0.5)       # 歇 0.5 秒再调 API，防限流
        response = self.kimi.chat_completion(     # 调用 Kimi API（有图时自动图文混传）
            prompt=full_prompt,
            temperature=1.0,
            max_tokens=8192,
            images=images
        )
        logger.info(f"论文分析完成: {paper_id}")
        return response


    def answer_question(self, paper_id: str, paper_text: str, question: str,
                        images: Optional[List[str]] = None) -> Tuple[str, List[str]]:
        """回答用户关于论文的问题（支持传入图片做多模态分析）"""
        logger.info(f"回答问题: {question[:50]}...")
        paper_text = self._strip_references(paper_text)  # 去掉参考文献条目，省 token
        full_prompt = f"{QA_PROMPT}\n\n论文内容：\n{paper_text[:190000]}\n\n用户问题：{question}"
        time.sleep(0.5)  # 歇 0.5 秒再调 API，防限流
        answer = self.kimi.chat_completion(       # 调用 Kimi API（有图时自动图文混传）
            prompt=full_prompt,
            temperature=1.0,
            max_tokens=4096,
            images=images
        )
        return answer, self._extract_sources(answer)


    @staticmethod
    def _strip_references(text: str) -> str:
        """去掉参考文献条目，保留 References 标题和后面的附录内容"""
        # 找 References / Bibliography / 参考文献 的起始位置
        ref_match = re.search(r'(?:^|\n)(References|REFERENCES|Bibliography|BIBLIOGRAPHY|参考文献)\s*\n', text, re.IGNORECASE)
        if not ref_match:
            return text  # 没找到参考文献区，原样返回

        ref_start = ref_match.start() + 1  # References 标题行的开头（跳过前面的换行）
        before_ref = text[:ref_start]       # References 前面的正文
        after_ref = text[ref_start:]        # References 标题行 + 条目 + 后续内容

        # 策略1：尝试匹配带方括号的引用 [1] 或 [Author, Year]
        cleaned = re.sub(r'^\[[^\]]+\][\s\S]*?\n(?:\n|Figure|Table|Appendix|\Z)', '\n', after_ref, flags=re.MULTILINE)

        # 如果策略1没删掉任何行（说明是纯文本格式的参考文献），直接截断
        if cleaned == after_ref:
            return before_ref.strip()

        # 去掉连在一起的多个空行，保留一个
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

        return (before_ref + cleaned).strip()


    def _extract_sources(self, answer: str) -> list:
        """提取引用来源"""
        sources = []
        for pattern in [r"根据.*?第\s*(\d+)\s*页", r"如图\s*(\d+)\s*所示", r"表\s*(\d+)\s*中", r"公式\s*\(?(\d+)\)?"]:
            for m in re.finditer(pattern, answer):
                sources.append(m.group(0))
        return list(set(sources))[:10]


# 单例
_analyzer: Optional[PaperAnalyzer] = None


def get_analyzer() -> PaperAnalyzer:
    """获取分析器单例"""
    global _analyzer
    if _analyzer is None:
        _analyzer = PaperAnalyzer()
    return _analyzer