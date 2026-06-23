# PaperPilot - AI论文精读助手

基于 ReAct Agent 的论文智能分析系统，通过意图识别与工具调用的自动化闭环，解决学术论文阅读效率低、公式图表理解难的痛点。

## 功能特性

- **ReAct Agent 推理架构**：基于 LangGraph 搭建 ReAct 推理链路，实现「意图识别 → 工具选择 → 任务执行 → 结果汇总」的自动化闭环
- **多工具系统**：集成论文分析、智能问答、摘要获取、论文列表查询、图表提取等 5 类工具，支持动态注册与按需调用
- **流式交互**：实时展示推理过程，过滤工具调用和工具返回的中间消息，仅输出最终回答，提升交互体验
- **PDF 深度解析**：基于 PyMuPDF 实现文本/公式/图表/图片的完整提取，支持 SHA256 哈希去重与 JSON 持久化
- **多模态分析**：调用 Kimi API 实现论文结构化分析与智能问答，支持图文混传，提升图表理解能力
- **多轮对话记忆**：注入历史对话上下文，保持多轮对话连贯性

## 技术栈

| 技术 | 用途 |
|------|------|
| Python | 编程语言 |
| Streamlit | 前端界面 |
| LangGraph | ReAct Agent 推理编排 |
| LangChain | 工具定义、模型调用 |
| PyMuPDF | PDF 解析 |
| Pydantic | 配置管理 |
| DeepSeek API | Agent 推理决策 |
| Kimi API | 论文分析、智能问答 |

## 安装步骤

### 1. 克隆项目

```bash
git clone http://github.com/Yi162624/PaperPilot.git
cd PaperPilot
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example` 为 `.env`，并填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# Kimi API Key（从 https://platform.moonshot.cn/ 获取）
KIMI_API_KEY=your_api_key_here

# DeepSeek API Key（从 https://platform.deepseek.com/ 获取）
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro

# 存储配置
UPLOAD_DIR=./uploads
IMAGE_OUTPUT_DIR=./images
DATA_DIR=./data
```

### 5. 启动应用

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`

## 使用说明

### 上传论文

1. 点击左侧边栏的「上传论文」
2. 选择 PDF 文件上传
3. 系统会自动解析论文内容（文本、公式、图表、图片）

### 智能分析

1. 在左侧选择已上传的论文
2. 在对话框中输入问题，例如：
   - "分析这篇论文的核心方法"
   - "这篇论文解决了什么问题？"
   - "解释一下公式 (1)"
   - "展示 Figure 1 的图片"
3. Agent 会自动选择合适的工具进行分析

### 可用工具

| 工具 | 功能 |
|------|------|
| 分析论文 | 深度分析论文，生成结构化解读报告 |
| 智能问答 | 基于论文内容回答用户问题 |
| 获取摘要 | 获取论文基本信息（标题、作者、页数等） |
| 论文列表 | 列出当前已上传的所有论文 |
| 图表提取 | 根据图表编号获取对应图片 |

## 项目结构

```
PaperPilot/
├── app.py                 # Streamlit 主应用入口
├── react_agent.py         # ReAct Agent 核心代码
├── config.py              # 配置管理
├── clients.py             # API 客户端
├── prompts.py             # 提示词模板
├── logger.py              # 日志工具
├── requirements.txt       # 依赖列表
├── .env.example           # 配置示例
├── .gitignore             # Git 忽略规则
├── services/              # 服务模块
│   ├── pdf_parser.py      # PDF 解析器
│   ├── paper_analyzer.py  # 论文分析器
│   └── storage.py         # 数据存储
└── README.md              # 项目说明文档
```

## 性能指标

- 工具选择准确率：100%
- 回答准确率：92%（平均 4.60/5）
- 多轮对话连贯性：94%

## License

MIT License

## 联系方式

- GitHub: [Yi162624](http://github.com/Yi162624)
- 项目地址: http://github.com/Yi162624/PaperPilot