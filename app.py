"""
PaperPilot - AI论文精读助手
Streamlit 前后端融合入口
启动命令: streamlit run app.py
"""

import streamlit as st
import uuid
from time import sleep
from pathlib import Path

from services.storage import get_papers_storage, add_paper, get_paper, delete_paper, calculate_file_hash, get_paper_by_hash
from services.pdf_parser import get_pdf_parser
from react_agent import get_agent


st.set_page_config(
    page_title="PaperPilot - AI论文精读助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_state():
    """初始化 session_state"""
    if "papers" not in st.session_state:
        st.session_state["papers"] = get_papers_storage()  # 获取论文存储字典（内存 + JSON）
    if "current_paper_id" not in st.session_state:
        st.session_state["current_paper_id"] = None
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "agent" not in st.session_state:
        st.session_state["agent"] = get_agent()  # 获取 ReAct Agent 单例
    if "is_processing" not in st.session_state:
        st.session_state["is_processing"] = False
    if "pending_prompt" not in st.session_state:
        st.session_state["pending_prompt"] = None


def handle_upload(uploaded_file):
    """处理 PDF 上传"""
    parser = get_pdf_parser()  # 获取 PDF 解析器单例

    file_bytes = uploaded_file.getvalue()
    file_hash = calculate_file_hash(file_bytes)  # 计算文件 SHA256 哈希，用于去重

    existing_id, existing_paper = get_paper_by_hash(file_hash)  # 根据哈希查重，已存在则跳过上传
    if existing_id:
        st.warning(f"该论文已存在：{existing_paper.get('title', '未知')}")
        st.session_state["current_paper_id"] = existing_id
        return existing_id

    file_path = parser.save_file(file_bytes, uploaded_file.name)  # 保存 PDF 到 uploads/ 目录

    with st.spinner("正在解析论文..."):
        result = parser.parse_paper(file_path)  # 解析 PDF：提取文本/公式/图表/章节

    paper_id = str(uuid.uuid4())
    paper_data = {
        "paper_id": paper_id,
        "title": result["title"],
        "authors": result["authors"],
        "text": result["full_text"],
        "formulas": result["formulas"],
        "figures": result["figures"],
        "tables": result["tables"],
        "sections": result["sections"],
        "images": result.get("images", []),            # 提取的图片列表
        "figure_image_map": result.get("figure_image_map", {}),  # 图表编号 → 图片路径映射
        "num_pages": result["num_pages"],
        "file_path": file_path,
        "file_hash": file_hash,
        "analyzed": False,
        "report": None
    }

    add_paper(paper_id, paper_data)  # 添加论文到存储并持久化到 JSON
    st.session_state["papers"] = get_papers_storage()  # 刷新论文列表
    st.session_state["current_paper_id"] = paper_id

    st.success(f"论文上传成功！\n\n**标题**: {result['title']}\n**作者**: {', '.join(result['authors']) or '未知'}\n**页数**: {result['num_pages']}")
    return paper_id


def main():
    init_state()

    st.title("📚 PaperPilot - AI论文精读助手")
    st.markdown("帮助计算机/AI专业学生理解复杂论文内容、解析公式图表、回答论文相关问题")

    # ========== 侧边栏 ==========
    with st.sidebar:
        st.header("📤 论文上传")
        # 上传论文
        uploaded_file = st.file_uploader("上传 PDF 论文", type="pdf", help="支持最大 50MB 的 PDF 文件")
        if uploaded_file:
            # 处理上传
            handle_upload(uploaded_file)

        st.divider()

        st.header("📚 论文列表")
        papers = st.session_state["papers"]

        if papers:
            # 显示论文列表
            paper_options = {p.get("title", "未知论文"): pid for pid, p in papers.items()}
            selected_titles = list(paper_options.keys())

            current_idx = 0
            if st.session_state["current_paper_id"]:
                for i, pid in enumerate(paper_options.values()):
                    if pid == st.session_state["current_paper_id"]:
                        current_idx = i
                        break

            selected_title = st.selectbox(
                "选择当前论文",
                options=selected_titles,
                index=current_idx
            )
            if selected_title:
                st.session_state["current_paper_id"] = paper_options[selected_title]
        else:
            st.info("暂无论文，请先上传 PDF")

        st.divider()

        st.header("📊 当前论文详情")
        current_paper_id = st.session_state["current_paper_id"]
        if current_paper_id:
            paper = get_paper(current_paper_id)  # 获取单篇论文的完整数据
            if paper:
                st.markdown(f"**标题**: {paper.get('title', '未知')}")
                st.markdown(f"**作者**: {', '.join(paper.get('authors', [])) or '未知'}")
                st.markdown(f"**页数**: {paper.get('num_pages', 0)}")
                st.markdown(f"**公式数**: {len(paper.get('formulas', []))}")
                st.markdown(f"**图表数**: {len(paper.get('figures', []))}")
                st.markdown(f"**分析状态**: {'✅ 已分析' if paper.get('analyzed') else '⏳ 未分析'}")

                if st.button("🗑️ 删除当前论文", use_container_width=True):
                    delete_paper(current_paper_id)  # 从存储中删除论文并持久化
                    st.session_state["papers"] = get_papers_storage()  # 刷新论文列表
                    st.session_state["current_paper_id"] = None
                    st.success("论文已删除")
                    st.rerun()
        else:
            st.info("请先选择或上传一篇论文")

        st.divider()

        if st.button("🗑️ 清空对话历史", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

    # ========== 主区域 - 聊天界面 ==========
    st.header("💬 智能问答")

    if not st.session_state["current_paper_id"]:
        st.warning("⚠️ 请先在左侧上传并选择论文，再开始提问")

    # 展示历史对话（按存储的顺序回放文字和图片）
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            # 新格式：按 output_items 顺序渲染
            if "output_items" in message:
                for item_type, item_content in message["output_items"]:
                    if item_type == "text":
                        st.markdown(item_content)
                    elif item_type == "image":
                        if Path(item_content).exists():
                            st.image(item_content)
            else:
                # 兼容旧格式（只有文本和图片分离的字段）
                st.write(message.get("content", ""))
                for img_path in message.get("images", []):
                    if Path(img_path).exists():
                        st.image(img_path)

    # 用户输入
    prompt = st.chat_input("请输入您的问题...", disabled=st.session_state["is_processing"])

    # 用户提交了问题且不在处理中 → 挂起问题，重绘画布，让输入框禁用
    if prompt and not st.session_state["is_processing"]:
        # 挂起用户问题
        st.session_state["pending_prompt"] = prompt
        # 开始处理用户输入
        st.session_state["is_processing"] = True
        st.rerun()

    # 处于处理中且有挂起的问题 → 执行真正的处理
    if st.session_state["is_processing"] and st.session_state["pending_prompt"]:
        prompt = st.session_state["pending_prompt"]
        st.session_state["pending_prompt"] = None

        # 处理用户输入
        if not st.session_state["current_paper_id"]:
            st.error("请先上传并选择论文！")
            st.session_state["is_processing"] = False
            st.rerun()
            return

        # 显示用户消息
        st.chat_message("user").write(prompt)
        # 保存用户消息到对话历史
        st.session_state["messages"].append({"role": "user", "content": prompt})

        response_messages = []

        try:
            with st.spinner("智能助手思考中..."):
                history = st.session_state["messages"][:-1]
                paper_id = st.session_state["current_paper_id"]

                # 调用 Agent 流式生成回答
                res_stream = st.session_state["agent"].execute_stream(
                    prompt,                # 用户问题
                    paper_id=paper_id,     # 当前论文ID
                    history=history        # 历史对话记录
                )

                # 按顺序收集所有输出（保留文字和图片的先后顺序）
                output_items = []  # [(type, content), ...]
                for item in res_stream:
                    output_items.append(item)

                # 在一个聊天气泡内按顺序渲染：文字和图片交替出现
                with st.chat_message("assistant"):
                    for item_type, item_content in output_items:
                        if item_type == "text":
                            st.markdown(item_content)
                        elif item_type == "image":
                            if Path(item_content).exists():
                                st.image(item_content)
                            else:
                                st.warning(f"图片文件不存在: {item_content}")

                # 保存到对话历史（包含顺序信息，方便回放）
                st.session_state["messages"].append({
                    "role": "assistant",
                    "output_items": output_items
                })
        finally:
            st.session_state["is_processing"] = False
            st.rerun()


if __name__ == "__main__":
    main()
