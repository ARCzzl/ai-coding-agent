# 领克汽车 RAG 智能销售顾问 🚗

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://www.langchain.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 基于 RAG + Agent 的智能汽车销售顾问，支持混合检索、工具调用、多轮对话和量化评估。

**面试/简历项目 | 本科实习 | 大模型算法工程师方向**

---

## 🎯 项目亮点

- **三层检索 Pipeline**：Dense + Sparse 混合检索 → Cross-Encoder 精排 → Multi-Query 查询重写
- **Agent 工具调用**：LLM 自主决策调用知识检索 / 贷款计算 / 车型对比
- **RAGAS 评估体系**：4 维指标量化系统效果，消融实验驱动优化
- **FastAPI 服务化**：RESTful API + SSE 流式输出 + Swagger 文档
- **模块化架构**：7 个模块 15 个文件，清晰的分层设计

## 🏗️ 系统架构

```
用户
 ↓
FastAPI (SSE 流式输出)
 ↓
Agent (LangChain Tool Calling)
 ├── RAG 检索工具 ──→ 三层检索 Pipeline
 │                    ├── Multi-Query 重写
 │                    ├── 混合检索 (Dense + Sparse)
 │                    └── Cross-Encoder 重排序
 ├── 贷款计算工具 ──→ 等额本息计算
 └── 车型对比工具 ──→ 价格 + 定位对比
 ↓
LLM (通义千问 qwen3.6-plus)
 ↑
对话记忆 (Buffer / Window / Summary)
```

## 📁 项目结构

```
├── config/              # 配置管理（环境变量）
│   └── settings.py
├── core/                # 核心 RAG 组件
│   ├── loader.py        # 文档加载与清洗
│   ├── chunker.py       # 3种分块策略
│   ├── retriever.py     # 混合检索 + 查询重写
│   ├── reranker.py      # Cross-Encoder 重排序
│   ├── llm.py           # LLM 封装
│   └── memory.py        # 3种对话记忆策略
├── agent/               # Agent 层
│   ├── agent.py         # 智能销售顾问 Agent
│   └── tools.py         # 工具定义（检索/计算/对比）
├── api/                 # Web 服务
│   └── server.py        # FastAPI + SSE
├── evaluation/          # 评估体系
│   ├── metrics.py       # RAGAS 4维指标
│   └── run_eval.py      # 评估执行脚本
├── experiments/         # 实验记录
│   └── experiment_log.md
├── app.py               # CLI 入口
├── run_api.py           # API 入口
├── INTERVIEW_GUIDE.md   # 面试准备指南
├── RESUME.md            # 简历项目描述
└── requirements.txt
```

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/你的用户名/lynkco-rag-agent.git
cd lynkco-rag-agent
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 LLM_API_KEY
```

### 4. 准备数据

将 `car_info.pdf` 放入项目根目录。

### 5. 启动服务

```bash
# CLI 交互模式
python app.py

# 运行评估
python app.py --eval

# API 服务模式
python run_api.py
# 访问 http://localhost:8000/docs 查看 Swagger 文档
```

## 📊 评估指标

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| Recall@5 | 0.60 | 0.82 | +36% |
| Faithfulness | 0.78 | 0.88 | +10pp |
| MRR | 0.45 | 0.70 | +56% |

详细实验记录见 [experiments/experiment_log.md](experiments/experiment_log.md)

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| LLM | 通义千问 qwen3.6-plus（DashScope） |
| Embedding | qwen3-embedding:0.6b（Ollama 本地部署） |
| Reranker | BAAI/bge-reranker-large |
| 框架 | LangChain · FastAPI |
| 向量数据库 | ChromaDB |
| 评估 | RAGAS（Faithfulness/Relevancy/Recall/Precision） |
| NLP | jieba 分词 · PyMuPDF 解析 |

## 📖 更多文档

- [面试准备指南](INTERVIEW_GUIDE.md) — 高频面试问题与回答话术
- [简历项目描述](RESUME.md) — 三种风格的简历写法
- [实验记录](experiments/experiment_log.md) — 5 组消融实验数据

## 📄 License

MIT License