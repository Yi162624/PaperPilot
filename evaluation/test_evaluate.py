"""
Agent 效果评估脚本
用 DeepSeek 做裁判，评估 ReAct Agent 的三项指标：
  1. 工具选择准确率
  2. 回答准确率（LLM-as-Judge 打分）
  3. 多轮对话连贯性
用法: cd evaluation && python test_evaluate.py --qa_file qa/transformer_qa.json --paper_id <论文ID>
"""

import json
import logging
import re
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# 加父目录到 sys.path，让 evaluation/ 子目录能导入 paper-pilot 的模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler

from config import get_settings
from services.storage import get_paper
from react_agent import get_agent, PaperPilotAgent
from logger import logger


# ============================================================
# 评估专用日志器 — 每一步都写到 evaluation/logs/ 下
# ============================================================

class EvalLogger:
    """评估专用日志，同时输出到控制台和文件"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self._log = logging.getLogger("paperpilot_eval")
        self._log.setLevel(logging.DEBUG)
        if self._log.handlers:
            return

        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 文件 — 写到 evaluation/logs/ 下，每次一个带时间戳的文件
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self._log.addHandler(fh)
        self._log_file = str(log_file)

        # 控制台 — 只输出关键信息
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self._log.addHandler(ch)

    def get_log_file(self) -> str:
        return self._log_file

    def step_start(self, msg: str):
        self._log.info(f"================================>{msg}")

    def step_end(self, msg: str):
        self._log.info(f"<================================{msg}")

    def info(self, msg: str):
        self._log.info(msg)

    def debug(self, msg: str):
        self._log.debug(msg)

    def warn(self, msg: str):
        self._log.warning(msg)

    def error(self, msg: str):
        self._log.error(msg)

    def agent_answer(self, qid, question: str, answer: str, tools: list):
        """记录 Agent 的完整问答"""
        self._log.info(f"[Q{qid}] 问题: {question}")
        self._log.info(f"[Q{qid}] 工具: {tools}")
        self._log.info(f"[Q{qid}] 回答:\n{answer}")
        self._log.info(f"[Q{qid}] 回答长度: {len(answer)} 字符")

    def judge_raw(self, qid, raw_text: str):
        """记录裁判原始返回"""
        self._log.debug(f"[Q{qid}] 裁判原始返回:\n{raw_text}")

    def judge_result(self, qid, score: float, reason: str):
        """记录裁判打分结果"""
        self._log.info(f"[Q{qid}] 裁判打分: {score:.2f}/5 | 理由: {reason}")

    def multi_round(self, group_name: str, round_num: int, question: str, answer: str):
        """记录多轮对话每一轮"""
        self._log.info(f"[多轮:{group_name}] 第{round_num}轮问题: {question}")
        self._log.info(f"[多轮:{group_name}] 第{round_num}轮回答:\n{answer}")

    def api_error(self, context: str, error_msg: str):
        """记录 API 错误"""
        self._log.error(f"[API错误] {context}: {error_msg}")


_eval_log: Optional[EvalLogger] = None


def get_eval_log() -> EvalLogger:
    """获取评估日志器单例"""
    global _eval_log
    if _eval_log is None:
        _eval_log = EvalLogger()
    return _eval_log


# ============================================================
# 工具调用追踪器 — 记录 Agent 实际调了哪些工具
# ============================================================

class ToolTracker(BaseCallbackHandler):
    """LangChain 回调：捕获每次工具调用"""

    def __init__(self):
        super().__init__()
        self.called_tools: List[str] = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "unknown")
        self.called_tools.append(name)

    def reset(self):
        self.called_tools = []


# ============================================================
# DeepSeek 裁判 — 用 DeepSeek API 给 Agent 回答打分
# ============================================================

class DeepSeekJudge:
    """用 DeepSeek 模型做 LLM-as-Judge"""

    def __init__(self):
        cfg = get_settings()
        self.client = OpenAI(
            api_key=cfg.deepseek_api_key,
            base_url=cfg.deepseek_base_url
        )
        self.model = cfg.deepseek_model

    def score_single(self, question: str, ground_truth: str, agent_answer: str) -> Tuple[float, str]:
        """单轮问答评分：0-5 分 + 理由"""
        prompt = f"""你是一个专业的论文问答评估裁判。请根据以下信息对 AI 助手的回答质量打分（0-100分，分数越细越好，不要只打整数）。

【问题】：
{question}

【标准答案参考】：
{ground_truth}

【AI 回答】：
{agent_answer}

打分标准（100分 = 原5分，80分 = 原4分，依此类推，但请打出更细的具体分数如 87、92 等）：
- 90-100分：核心信息完全正确，且是论文原文中有的内容；如果问题包含错误前提，先明确指出前提错误再给出正确信息
- 80-89分：核心信息正确，但有小遗漏或表述不够精确；错误前提题纠正了但表述不够利落
- 70-79分：核心信息基本正确，只遗漏了少量次要细节
- 60-69分：核心信息基本正确，但遗漏了一处重要细节或表述不够到位；错误前提题没有明确指出，但给了正确信息
- 50-59分：大部分信息正确，但有一两个关键点答错或遗漏
- 40-49分：只有部分信息正确，多个关键点答错或遗漏严重；错误前提题顺着错误前提编造了部分内容
- 20-39分：大部分信息错误，基本没答到点上
- 0-19分：没有给出有效回答，或者完全脱离论文内容、纯靠自己的常识编造，或者完全没有回答问题

请严格只输出下面两行，不要有任何开头语、不要复述问题、不要分析过程，全文不超过 700 字：
分数: XX
理由: （一句话说明得分或扣分原因，不超过 100 字）"""

        return self._divide_score(self._call_judge(prompt))

    def score_multi_turn(self, group_name: str, rounds: List[Dict],
                         agent_responses: List[str], ground_truth: str) -> Tuple[float, str]:
        """多轮对话评分：评估各轮回答之间的连贯性"""
        rounds_text = ""
        for i, (rd, resp) in enumerate(zip(rounds, agent_responses)):
            rounds_text += f"第{i+1}轮-问题：{rd['question']}\n第{i+1}轮-AI回答：{resp}\n\n"

        prompt = f"""你是一个专业的多轮对话评估裁判。请评估以下多轮对话的连贯性（0-100分，分数越细越好，不要只打整数）。

【对话主题】：{group_name}

【对话过程】：
{rounds_text}

【标准答案参考】：
{ground_truth}

打分标准（100分 = 原5分，80分 = 原4分，依此类推，但请打出更细的具体分数如 87、92 等）：
- 90-100分：所有轮次回答高度连贯，AI 正确引用了前轮信息，逻辑递进自然流畅
- 80-89分：回答基本连贯，能引用前轮信息但偶尔不够完美
- 70-79分：大部分轮次连贯，只有个别轮次衔接略显生硬
- 60-69分：部分轮次连贯，但有一两轮显得独立，前后关联较弱
- 50-59分：只有部分轮次之间有呼应，整体连贯性偏弱
- 40-49分：各轮回答基本独立，缺乏多轮对话应有的连贯性
- 20-39分：完全看不出多轮对话的连贯性，各轮答非所问
- 0-19分：完全没有有效回答，或者所有轮次都是胡编乱造、与论文内容毫无关系

请严格只输出下面两行，不要有任何开头语、不要复述问题、不要分析过程，全文不超过 700 字：
分数: XX
理由: （说明多轮对话连贯性表现，不超过 100 字）"""

        return self._divide_score(self._call_judge(prompt))

    def _call_judge(self, prompt: str, retries: int = 2) -> Tuple[int, str]:
        """调用 DeepSeek 打分（0-100），返回（原始分数, 理由），空响应自动重试"""
        text = None
        base_delay = 3  # 基础等待秒数，指数增长避开限流窗口
        evl = get_eval_log()

        for attempt in range(retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,  # 0.1 避免空响应，对打分波动 < ±0.1 分
                    max_tokens=800  # 给足输出空间，防止回答太长时被截断
                )
                text = response.choices[0].message.content.strip()
                if text:  # 非空就跳出重试
                    evl.info(f"裁判返回({len(text)}字符)")
                    evl.debug(f"裁判原始文本:\n{text}")
                    break
                if attempt < retries:
                    delay = base_delay * (2 ** attempt)  # 3s → 6s → 12s
                    evl.warn(f"裁判返回空内容，{delay}秒后第{attempt+1}次重试...")
                    logger.warning(f"裁判返回空内容，{delay}秒后第{attempt+1}次重试...")
                    time.sleep(delay)
            except Exception as e:
                evl.api_error("DeepSeek 裁判调用", str(e))
                logger.error(f"DeepSeek 裁判调用失败: {e}")
                if attempt < retries:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    return 60, f"裁判调用异常: {e}"

        if not text:
            logger.warning(f"裁判{retries+1}次均返回空，回退默认分")
            return 60, "裁判返回空内容"

        # 多层正则匹配，从严格到宽松
        score = None
        reason = None

        # 第 1 层：标准格式 "分数: 85" / "得分: 92" / "总分：78"
        score_match = re.search(r'(?:分数|得分|总分)\s*[:：]\s*(\d+)', text)
        reason_match = re.search(r'理由[:：]\s*(.+)', text, re.DOTALL)

        # 第 2 层：容错 — 用换行分隔，找分数行和理由行
        if not score_match or not reason_match:
            lines = text.split('\n')
            for i, line in enumerate(lines):
                m = re.search(r'(?:分数|得分|总分)\s*[:：]?\s*(\d+)', line)
                if m and score_match is None:
                    score_match = m
                m2 = re.search(r'理由[:：]?\s*(.+)', line)
                if m2 and reason_match is None:
                    reason_match = m2

        # 第 3 层：兜底 — 在全文里捞 0-100 之间的数字作为分数
        if not score_match:
            all_nums = re.findall(r'\b(\d{1,3})\b', text)
            valid_scores = [int(n) for n in all_nums if 0 <= int(n) <= 100]
            if valid_scores:
                # 取最后一个出现的有效数字（通常是分数）
                score = valid_scores[-1]

        if score_match:
            score = int(score_match.group(1))
        if reason_match:
            reason = reason_match.group(1).strip()

        # 都匹配不到时记完整日志
        if score is None or not reason:
            logger.warning(f"裁判返回格式异常，原始文本: {text}")

        score = score if score is not None else 60
        score = max(0, min(100, score))
        reason = reason if reason else "无法解析评分理由"
        return score, reason

    @staticmethod
    def _divide_score(result: Tuple[int, str]) -> Tuple[float, str]:
        """把 0-100 分除以 20 还原为 0-5 分，保留两位小数"""
        raw_score, reason = result
        final_score = round(raw_score / 20.0, 2)
        return final_score, reason


# ============================================================
# Agent 调用封装 — 带工具追踪
# ============================================================

def run_agent_with_tracking(query: str, paper_id: str,
                            history: Optional[List[Dict]] = None) -> Tuple[str, List[str]]:
    """
    调用 Agent 并追踪工具使用
    返回: (Agent 完整回答文本, 实际调用的工具名列表)
    """
    agent = get_agent()
    tracker = ToolTracker()

    # 复刻 execute_stream 的消息构建逻辑，加上 callback
    messages = []
    if history:
        for msg in history[-10:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
    if paper_id:
        messages.append(SystemMessage(content=f"【当前论文ID】：{paper_id}"))
    messages.append(HumanMessage(content=query))

    input_dict = {"messages": messages}

    full_text = ""
    try:
        for chunk in agent.agent.stream(
            input_dict,
            stream_mode="values",
            config={"callbacks": [tracker]}
        ):
            latest = chunk["messages"][-1]
            # 跳过工具调用消息（AI 推理"该调哪个工具"）
            if hasattr(latest, "tool_calls") and latest.tool_calls:
                continue
            # 跳过工具返回消息（工具拿到的原始数据）
            if type(latest).__name__ == "ToolMessage":
                continue
            if latest.content:
                # 用 _parse_image_content 拆出纯文本
                for item_type, item_text in PaperPilotAgent._parse_image_content(latest.content.strip()):
                    if item_type == "text":
                        full_text += item_text
    except Exception as e:
        evl = get_eval_log()
        evl.api_error("Agent 执行", str(e))
        logger.error(f"Agent 执行失败: {e}")
        full_text = f"[执行失败: {e}]"

    return full_text.strip(), tracker.called_tools


# ============================================================
# 报告生成
# ============================================================

def write_report(paper_title_cn: str, paper_title_en: str, paper_id: str,
                 single_results: List[Dict], multi_results: List[Dict]) -> str:
    """用 DeepSeek 生成 Markdown 评估报告，返回报告文件路径"""

    # 确保输出目录存在
    report_dir = Path(__file__).parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # 生成文件名
    safe_title = paper_title_cn.replace("/", "_").replace("\\", "_").replace(" ", "_")
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{safe_title}_{date_str}.md"
    report_path = report_dir / report_filename

    # ---- 计算指标 ----
    total_tool_checks = len(single_results)
    tool_correct = sum(1 for r in single_results if r["expected_tool"] in r["actual_tools"])
    tool_accuracy = (tool_correct / total_tool_checks * 100) if total_tool_checks > 0 else 0

    all_scores = [r["score"] for r in single_results] + [r["score"] for r in multi_results]
    answer_accuracy = (sum(all_scores) / (len(all_scores) * 5) * 100) if all_scores else 0

    multi_scores = [r["score"] for r in multi_results]
    multi_ability = (sum(multi_scores) / (len(multi_scores) * 5) * 100) if multi_scores else 0

    # ---- 构建给 DeepSeek 的数据 ----
    single_lines = ""
    for r in single_results:
        tool_ok = "OK" if r["expected_tool"] in r["actual_tools"] else "MISS"
        single_lines += f"- #{r['id']} | 问题: {r['question']} | 得分: {r['score']:.2f}/5 | 理由: {r['reason']} | 工具: {tool_ok}\n"

    multi_lines = ""
    for mr in multi_results:
        multi_lines += f"- {mr['group_name']} ({len(mr['rounds'])}轮) | 连贯性: {mr['score']:.2f}/5 | 理由: {mr['reason']}\n"

    # ---- 调 DeepSeek 生成报告 ----
    cfg = get_settings()
    client = OpenAI(api_key=cfg.deepseek_api_key, base_url=cfg.deepseek_base_url)

    prompt = f"""你是专业的评估报告撰写助手。请根据以下数据生成一份简洁的 Markdown 评估报告。不要添加任何多余内容。

【论文】：{paper_title_cn}（{paper_title_en}）
【评估时间】：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
【裁判模型】：DeepSeek（{cfg.deepseek_model}）

【综合指标】：
- 工具选择准确率: {tool_accuracy:.1f}%（{tool_correct}/{total_tool_checks}）
- 回答准确率: {answer_accuracy:.1f}%（{len(all_scores)}题平均 {sum(all_scores)/len(all_scores):.2f}/5）
- 多轮对话能力: {multi_ability:.1f}%（{len(multi_scores)}组平均 {sum(multi_scores)/len(multi_scores):.2f}/5）

【单轮问答结果】：
{single_lines}

【多轮对话结果】：
{multi_lines}

请按以下结构输出报告（直接输出 Markdown，不要加代码块包裹）：

# Agent 效果评估报告

**论文**：xxx
**评估时间**：xxx
**裁判模型**：xxx

---

## 综合得分

（三行表格：指标 | 得分 | 说明）

---

## 单轮问答结果

（表格：序号 | 问题 | 分数 | 评分理由）

---

## 多轮对话结果

（表格：主题 | 轮数 | 连贯性 | 理由）

---

## 主要发现

（3-5条关键结论）"""

    try:
        response = client.chat.completions.create(
            model=cfg.deepseek_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        report_content = response.choices[0].message.content.strip()
        # 去掉可能的 ```markdown ``` 包裹
        report_content = re.sub(r'^```(?:markdown)?\s*\n', '', report_content)
        report_content = re.sub(r'\n```\s*$', '', report_content)
    except Exception as e:
        logger.warning(f"DeepSeek 生成报告失败: {e}，回退到硬拼接")
        report_content = _write_report_fallback(
            paper_title_cn, paper_title_en, paper_id,
            tool_accuracy, tool_correct, total_tool_checks,
            answer_accuracy, all_scores, multi_ability, multi_scores,
            single_results, multi_results
        )

    # 写入文件
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"评估报告已保存: {report_path}")
    return str(report_path)


def _write_report_fallback(paper_title_cn, paper_title_en, paper_id,
                           tool_accuracy, tool_correct, total_tool_checks,
                           answer_accuracy, all_scores, multi_ability, multi_scores,
                           single_results, multi_results) -> str:
    """DeepSeek 失败时的兜底报告（硬拼接）"""
    lines = []
    lines.append(f"# Agent 效果评估报告")
    lines.append("")
    lines.append(f"**论文**：{paper_title_cn}（{paper_title_en}）")
    lines.append(f"**评估时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**裁判模型**：DeepSeek（`{get_settings().deepseek_model}`）")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 综合得分")
    lines.append("")
    lines.append("| 指标 | 得分 | 说明 |")
    lines.append("|------|:---:|------|")
    lines.append(f"| 工具选择准确率 | **{tool_accuracy:.1f}%** | {tool_correct}/{total_tool_checks} 次选对工具 |")
    lines.append(f"| 回答准确率 | **{answer_accuracy:.1f}%** | {len(all_scores)} 题平均 {sum(all_scores)/len(all_scores):.2f}/5 分 |")
    lines.append(f"| 多轮对话能力 | **{multi_ability:.1f}%** | {len(multi_scores)} 组平均 {sum(multi_scores)/len(multi_scores):.2f}/5 分 |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 单轮问答结果")
    lines.append("")
    lines.append("| # | 问题 | 分数 | 理由 |")
    lines.append("|:-:|------|:---:|------|")
    for r in single_results:
        lines.append(f"| {r['id']} | {r['question'][:40]} | {r['score']:.2f} | {r['reason'][:60]} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 多轮对话结果")
    lines.append("")
    lines.append("| 主题 | 轮数 | 连贯性 | 理由 |")
    lines.append("|------|:---:|:---:|------|")
    for mr in multi_results:
        lines.append(f"| {mr['group_name']} | {len(mr['rounds'])} | {mr['score']:.2f}/5 | {mr['reason'][:60]} |")
    lines.append("")
    return "\n".join(lines)


# ============================================================
# 主流程
# ============================================================

def parse_args():
    """解析命令行参数 --qa_file --paper_id"""
    args = {}
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            key = arg.lstrip("--")
            args[key] = None
        elif "=" in arg and arg.startswith("--"):
            key, val = arg.lstrip("--").split("=", 1)
            args[key] = val
        else:
            # 上一个 key 的值
            for k in reversed(list(args.keys())):
                if args[k] is None:
                    args[k] = arg
                    break

    qa_file = args.get("qa_file", "qa/transformer_qa.json")
    paper_id = args.get("paper_id", None)

    return qa_file, paper_id


def main():
    # 强制 stdout 用 utf-8，避免中文/特殊字符打印报错
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    evl = get_eval_log()

    logger.info("=" * 50)
    logger.info("PaperPilot Agent 效果评估开始")
    logger.info("=" * 50)

    evl.step_start("评估启动")

    # 1. 解析参数
    qa_file_path, paper_id = parse_args()

    if not paper_id:
        logger.error("缺少 paper_id 参数，用法: python test_evaluate.py --qa_file xxx.json --paper_id <ID>")
        logger.info("当前存储中的论文：")
        from services.storage import get_papers_storage
        for pid, p in get_papers_storage().items():
            logger.info(f"  {pid[:8]}... | {p.get('title', '未知')}")
        return

    # 2. 加载 QA 数据 — 优先在当前目录找，找不到去脚本所在目录找
    qa_path = Path(qa_file_path)
    if not qa_path.exists():
        qa_path = Path(__file__).parent / qa_file_path  # 回退到 evaluation/ 目录下
    if not qa_path.exists():
        logger.error(f"QA 数据文件不存在: {qa_path}")
        return

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    paper_title_cn = qa_data.get("paper_title_cn", "未知论文")
    paper_title_en = qa_data.get("paper_title_en", "Unknown")
    single_turn_questions = qa_data.get("single_turn", [])
    multi_turn_groups = qa_data.get("multi_turn", [])

    logger.info(f"论文标题: {paper_title_cn} ({paper_title_en})")
    logger.info(f"单轮问题数: {len(single_turn_questions)}")
    logger.info(f"多轮对话组数: {len(multi_turn_groups)}")

    evl.info(f"论文: {paper_title_cn} ({paper_title_en})")
    evl.info(f"QA文件: {qa_path}")
    evl.info(f"单轮问题数: {len(single_turn_questions)} | 多轮组数: {len(multi_turn_groups)}")
    evl.info(f"评估日志: {evl.get_log_file()}")

    # 3. 加载论文数据
    paper = get_paper(paper_id)
    if not paper:
        logger.error(f"论文 {paper_id} 不存在")
        return

    logger.info(f"论文已加载: {paper.get('title', '未知')}, 文本长度: {len(paper.get('text', ''))}")

    # 4. 初始化裁判
    judge = DeepSeekJudge()
    logger.info("DeepSeek 裁判已就绪")

    evl.step_end("初始化完成")

    # 5. 逐个跑单轮测试
    logger.info("\n--- 开始单轮问答测试 ---")
    evl.step_start("单轮问答测试")
    single_results = []

    for qa in single_turn_questions:
        qid = qa["id"]
        question = qa["question"]
        expected_tool = qa["expected_tool"]
        ground_truth = qa["ground_truth"]

        evl.step_start(f"Q{qid} 开始: {question[:60]}")
        logger.info(f"[单轮 {qid}] {question[:40]}...")

        # 调 Agent
        agent_answer, actual_tools = run_agent_with_tracking(question, paper_id)
        evl.agent_answer(qid, question, agent_answer, actual_tools)

        # DeepSeek 打分
        time.sleep(0.5)  # 防 API 限流
        score, reason = judge.score_single(question, ground_truth, agent_answer)
        evl.judge_result(qid, score, reason)

        single_results.append({
            "id": qid,
            "question": question,
            "ground_truth": ground_truth,
            "expected_tool": expected_tool,
            "actual_tools": actual_tools,
            "agent_answer": agent_answer,
            "score": score,
            "reason": reason
        })

        logger.info(f"  实际工具: {actual_tools} | 期望: {expected_tool} | 得分: {score:.2f}/5")
        evl.step_end(f"Q{qid} 完成: 得分={score:.2f}/5")

        # 实时打印当前题目结果
        tool_ok = "ok" if expected_tool in actual_tools else "MISS"
        print(f"\n  [{qid}/15] {question[:50]}")
        print(f"    工具: {actual_tools} (期望={expected_tool}) [{tool_ok}]")
        print(f"    得分: {score:.2f}/5 — {reason}")
        print(f"    Agent 回答: {agent_answer[:200]}...")
        sys.stdout.flush()

    # 6. 跑多轮对话测试
    logger.info("\n--- 开始多轮对话测试 ---")
    evl.step_start("多轮对话测试")
    multi_results = []

    for group in multi_turn_groups:
        group_name = group.get("group_name", "未命名")
        rounds = group.get("rounds", [])
        ground_truth = group.get("ground_truth", "")

        evl.step_start(f"多轮: {group_name} ({len(rounds)}轮)")
        logger.info(f"[多轮] {group_name} ({len(rounds)} 轮)")

        agent_responses = []
        history = []

        for rd in rounds:
            question = rd["question"]
            rn = rd["round"]
            logger.info(f"  第 {rn} 轮: {question[:40]}...")

            time.sleep(0.5)
            agent_answer, _ = run_agent_with_tracking(question, paper_id, history=history)
            agent_responses.append(agent_answer)

            evl.multi_round(group_name, rn, question, agent_answer)

            # 更新对话历史，供下一轮使用
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": agent_answer})

        # DeepSeek 裁判对整组多轮对话打分
        time.sleep(0.5)
        score, reason = judge.score_multi_turn(group_name, rounds, agent_responses, ground_truth)
        evl.info(f"[多轮:{group_name}] 连贯性得分: {score:.2f}/5 | 理由: {reason}")

        multi_results.append({
            "group_name": group_name,
            "rounds": rounds,
            "agent_responses": agent_responses,
            "ground_truth": ground_truth,
            "score": score,
            "reason": reason
        })

        logger.info(f"  多轮连贯性得分: {score:.2f}/5")
        evl.step_end(f"多轮: {group_name} 完成: 连贯性={score:.2f}/5")

        # 实时打印多轮结果
        print(f"\n  [多轮] {group_name} ({len(rounds)} 轮)")
        print(f"    连贯性得分: {score:.2f}/5 — {reason}")
        sys.stdout.flush()

    # 7. 生成报告
    logger.info("\n--- 生成评估报告 ---")
    evl.step_start("生成报告")
    report_path = write_report(paper_title_cn, paper_title_en, paper_id, single_results, multi_results)
    evl.info(f"报告路径: {report_path}")
    evl.step_end("报告生成完成")

    # 8. 打印摘要
    total_tools = len(single_results)
    tool_correct = sum(1 for r in single_results if r["expected_tool"] in r["actual_tools"])
    all_scores = [r["score"] for r in single_results] + [r["score"] for r in multi_results]
    multi_scores = [r["score"] for r in multi_results]

    evl.step_start("评估摘要")
    evl.info(f"工具选择准确率: {tool_correct}/{total_tools} = {tool_correct/total_tools*100:.1f}%")
    evl.info(f"回答准确率: {sum(all_scores)/len(all_scores):.2f}/5 = {sum(all_scores)/(len(all_scores)*5)*100:.1f}%")
    if multi_scores:
        evl.info(f"多轮对话能力: {sum(multi_scores)/len(multi_scores):.2f}/5 = {sum(multi_scores)/(len(multi_scores)*5)*100:.1f}%")
    evl.info(f"详细报告: {report_path}")
    evl.step_end("评估摘要")

    print("\n" + "=" * 50)
    print("  评估完成 — 摘要")
    print("=" * 50)
    print(f"  工具选择准确率: {tool_correct}/{total_tools} = {tool_correct/total_tools*100:.1f}%")
    print(f"  回答准确率:     {sum(all_scores)/len(all_scores):.2f}/5 = {sum(all_scores)/(len(all_scores)*5)*100:.1f}%")
    if multi_scores:
        print(f"  多轮对话能力:   {sum(multi_scores)/len(multi_scores):.2f}/5 = {sum(multi_scores)/(len(multi_scores)*5)*100:.1f}%")
    print(f"\n  详细报告: {report_path}")
    print(f"  调试日志: {evl.get_log_file()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
