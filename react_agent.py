"""
ReAct Agent 模块
LangGraph ReAct Agent + 工具集 + 流式输出
使用 DeepSeek API + create_react_agent
"""

from typing import Optional, Dict, List, Tuple

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import get_settings
from services.paper_analyzer import get_analyzer
from services.storage import get_papers_storage, get_paper, update_paper
from prompts import AGENT_SYSTEM_PROMPT
from logger import logger


@tool(description="对指定的论文进行深入分析，生成结构化解读报告（背景、方法、公式、实验等）")
def analyze_paper_tool(paper_id: str) -> str:
    """分析论文工具"""
    analyzer = get_analyzer()  # 获取论文分析器单例
    paper = get_paper(paper_id)  # 从存储获取论文数据
    if not paper:
        return f"错误：论文 {paper_id} 不存在"
    try:
        # 提取图片路径列表（按路径去重），发给 Kimi 做多模态分析
        # images = list(dict.fromkeys(img["path"] for img in paper.get("images", []) if img.get("path")))
        report = analyzer.analyze_paper(paper_id, paper.get("text", ""))  # 调 Kimi API 深度分析论文
        update_paper(paper_id, {"analyzed": True, "report": report})  # 更新论文分析状态

        # 在 Kimi 报告中每个 Figure 描述后面紧跟原始图片
        import re as re_mod
        figure_map = paper.get("figure_image_map", {})

        def _attach_image(m):
            """找到 Figure N 对应的图片路径，拼成 [IMAGE:path]，没有匹配则原样返回"""
            fig_num = m.group(2)
            path = figure_map.get(f"Figure {fig_num}") or figure_map.get(fig_num)
            if path:
                return f"{m.group(0)}\n[IMAGE:{path}]"
            return m.group(0)

        # 每个 Figure 描述后面紧跟它的原图
        report = re_mod.sub(r'(Figure\s+(\d+[a-zA-Z]?)\s*[:：；;—–\-].*?)(?=\n\n|\n[A-Z#]|\nFigure|\nFig|\Z)', _attach_image, report, flags=re_mod.IGNORECASE)

        return f"分析完成，报告已生成。\n\n{report}"
    except Exception as e:
        logger.error(f"分析论文失败: {e}")
        return f"分析失败: {str(e)}"


@tool(description="基于论文内容回答用户的问题，用户正常使用智能问答问问题的时候调用")
def answer_question_tool(paper_id: str, question: str) -> str:
    """回答问题工具"""
    # 参考文献类问题直接拦截，不走 Kimi（提示词拦不住，必须在代码层挡）
    ref_keywords = ["参考文献", "引用", "citation", "reference", "bibliography",
                    "et al", "等人的工作", "引用来源"]
    if any(kw in question.lower() for kw in ref_keywords):
        return "此信息无法检索，请自行查看论文原文。"

    analyzer = get_analyzer()  # 获取论文分析器单例
    paper = get_paper(paper_id)  # 从存储获取论文数据
    if not paper:
        return f"错误：论文 {paper_id} 不存在"
    try:
        # 提取图片路径列表（按路径去重），发给 Kimi 做多模态分析
        # images = list(dict.fromkeys(img["path"] for img in paper.get("images", []) if img.get("path")))
        answer, sources = analyzer.answer_question(paper_id, paper.get("text", ""), question)  # 调 Kimi API 回答论文问题
        result = answer
        if sources:
            result += f"\n\n引用来源：{', '.join(sources)}"
        return result
    except Exception as e:
        logger.error(f"回答问题失败: {e}")
        return f"回答失败: {str(e)}"


@tool(description="获取论文的基本信息（标题、作者、页数、是否已分析）")
def get_paper_summary_tool(paper_id: str) -> str:
    """获取论文基本信息（同步，只读存储）"""
    paper = get_paper(paper_id)  # 从存储获取论文数据
    if not paper:
        return f"错误：论文 {paper_id} 不存在"
    return (
        f"标题：{paper.get('title', '未知')}\n"
        f"作者：{'、'.join(paper.get('authors', [])) or '未知'}\n"
        f"文本长度：{len(paper.get('text', ''))} 字符\n"
        f"是否已分析：{'是' if paper.get('analyzed') else '否'}"
    )


@tool(description="列出当前已上传的所有论文")
def list_papers_tool() -> str:
    """列出论文（同步，只读存储）"""
    papers = get_papers_storage()  # 获取所有论文存储字典
    if not papers:
        return "当前没有已上传的论文"
    lines = ["已上传的论文列表："]
    for pid, p in papers.items():
        analyzed = "[已分析]" if p.get("analyzed") else "[未分析]"
        lines.append(f"- {p.get('title', '未知')} ({analyzed}) ID: {pid[:8]}...")
    return "\n".join(lines)


@tool(description="根据图表编号获取论文中对应图片，用于展示给用户查看。传 'all' 可以获取所有图片（含 Figure 和 Table）。传纯数字如 '3' 表示第3个图表（按论文出现顺序）")
def get_figure_image_tool(paper_id: str, figure_number: str) -> str:
    """根据图表编号获取图片路径（支持 Figure、Table、序号查询）"""
    paper = get_paper(paper_id)
    if not paper:
        return f"错误：论文 {paper_id} 不存在"
    figure_map = paper.get("figure_image_map", {})       # 获取图表图片映射
    order_list = figure_map.get("_order", [])  # 按出现顺序排列的图表名列表

    # 传 "all" 时返回所有已配对的图表图片（Figure + Table，带序号）
    if figure_number.lower() == "all":
        if not order_list:
            return "未找到任何图表图片"
        result_parts = [f"已获取全部图片（共 {len(order_list)} 个图表）："]
        for i, key in enumerate(order_list, 1):
            result_parts.append(f"[IMAGE:{figure_map[key]}]\n{i}. {key}")
        return "\n".join(result_parts)

    # 纯数字 → 按论文中出现顺序的第 N 个图表（"3" = 第3个图表）
    if figure_number.isdigit():
        idx = int(figure_number) - 1
        if 0 <= idx < len(order_list):
            name = order_list[idx]
            return f"[IMAGE:{figure_map[name]}]\n已获取第 {figure_number} 个图表：{name}"
        return f"未找到第 {figure_number} 个图表，论文共有 {len(order_list)} 个图表"

    # 按 "Figure N"、"Table N" 两种名称查找
    image_path = (figure_map.get(f"Figure {figure_number}")
                  or figure_map.get(f"Table {figure_number}"))
    if not image_path:
        all_keys = [k for k in figure_map if k.startswith("Figure ") or k.startswith("Table ")]
        return f"未找到图表 {figure_number}，可用的图表有：{', '.join(all_keys) if all_keys else '无'}"
    return f"[IMAGE:{image_path}]\n已获取图表 {figure_number}"


class PaperPilotAgent:
    """论文精读 Agent"""

    def __init__(self):
        cfg = get_settings()  # 加载项目配置

        self.llm = ChatOpenAI(           # 初始化 DeepSeek API 模型
            api_key=cfg.deepseek_api_key,
            base_url=cfg.deepseek_base_url,
            model=cfg.deepseek_model,
            temperature=0.7
        )

        self.tools = [
            analyze_paper_tool,          # 分析论文工具
            answer_question_tool,        # 回答问题工具
            get_paper_summary_tool,      # 获取论文基本信息工具
            list_papers_tool,            # 列出论文工具
            get_figure_image_tool        # 获取论文图表图片工具
        ]

        self.agent = create_react_agent(    # 创建 LangGraph ReAct Agent
            model=self.llm,                 # 注入 DeepSeek 模型
            tools=self.tools,               # 注入工具集
            prompt=AGENT_SYSTEM_PROMPT      # 注入系统提示词
        )

        logger.info("Agent 初始化完成")

    def execute_stream(self, query: str, paper_id: Optional[str] = None, history: Optional[List[Dict]] = None):
        """
        流式执行 Agent

        Args:
            query: 用户当前输入的问题
            paper_id: 当前选中的论文 ID
            history: 历史对话记录，格式 [{"role": "user/assistant", "content": "..."}, ...]
        """
        messages = []
        if history:
            # 注入历史对话记录
            for msg in history[-10:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        if paper_id:
            # 注入当前论文ID
            messages.append({
                "role": "system",
                "content": f"【当前论文ID】：{paper_id}"
            })
        # 注入用户问题
        messages.append({"role": "user", "content": query})
        input_dict = {"messages": messages}

        logger.info(f"Agent 开始执行: {query[:50]}...")

        # 流式执行
        for chunk in self.agent.stream(input_dict, stream_mode="values"):
            latest = chunk["messages"][-1]
            # 跳过工具调用消息（AI 的中间推理"我先调工具"）
            if hasattr(latest, "tool_calls") and latest.tool_calls:
                continue
            # 工具返回消息：不直接展示，跳过。AI 会在最终回答里透传 [IMAGE:path] 标记
            if type(latest).__name__ == "ToolMessage":
                continue
            if latest.content:
                yield from self._parse_image_content(latest.content.strip())

        logger.info("Agent 执行完成")

    @staticmethod
    def _parse_image_content(content: str):
        """解析含 [IMAGE:path] 标记的内容，拆成文本和图片流"""
        import re as re_mod
        parts = re_mod.split(r'\[IMAGE:(.+?)\]', content)
        for i, part in enumerate(parts):
            if not part.strip():
                continue
            if i % 2 == 1:
                yield ("image", part.strip())  # 图片路径
            else:
                yield ("text", part.strip() + "\n")  # 文本

    def get_tools(self) -> List[Dict[str, str]]:
        """获取工具列表"""
        return [{"name": t.name, "description": t.description} for t in self.tools]


_agent: Optional[PaperPilotAgent] = None


def get_agent() -> PaperPilotAgent:
    """获取 Agent 单例"""
    global _agent
    if _agent is None:
        _agent = PaperPilotAgent()
    return _agent
