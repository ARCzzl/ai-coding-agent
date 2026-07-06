
**领克汽车 RAG 智能销售顾问系统** | 独立开发
*2025.06 - 2025.07*

- 设计并实现了一个基于 **RAG（检索增强生成）+ Agent** 的智能问答系统，以领克汽车产品手册 PDF 作为知识库，支持多轮对话的产品咨询、贷款计算和车型对比
- 构建了**三层检索 Pipeline**：混合检索（稠密向量 + BM25 稀疏检索 + jieba 中文分词）→ Cross-Encoder 重排序（BGE-reranker-large）→ 多查询重写（Multi-Query Rewriting），相比单一检索策略 Recall@5 提升 36%
- 基于 **LangChain + Ollama** 搭建 Agent 工具调用框架，集成知识检索、贷款计算、车型对比三个工具，实现 LLM 自主决策路由
- 建立 **RAGAS 评估体系**（忠实度、答案相关性、上下文召回率、上下文精确度 + Recall@K + MRR），通过消融实验量化每次策略调整的效果，忠实度从 0.78 提升至 0.88
- 使用 **FastAPI** 实现服务化部署，支持 SSE 流式输出和 Swagger 文档；设计 3 种对话记忆策略（Buffer/Window/Summary）适配不同对话场景

**技术栈**：Python · LangChain · ChromaDB · Ollama · FastAPI · BM25 · BGE-Reranker · jieba · PyMuPDF · RAGAS

---

## 版本二：精简版（适合一页简历，空间紧张时使用）

### 项目经历

**基于 RAG + Agent 的智能汽车销售顾问** | 独立开发 | *2025.06 - 2025.07*

- 设计并实现完整 RAG Pipeline（文档解析→智能分块→混合检索→重排序→LLM 生成），以领克汽车 PDF 手册为知识库，支持多轮对话的产品咨询和贷款计算
- 提出**三层检索策略**：Dense+Sparse 混合检索 + Cross-Encoder 精排 + Multi-Query 查询重写，Recall@5 达到 0.82
- 建立 RAGAS 评估体系量化系统效果，通过消融实验驱动优化，忠实度提升 10 个百分点
- 基于 LangChain Agent + FastAPI 完成服务化部署，支持流式输出和多策略对话记忆管理

**技术栈**：Python · LangChain · ChromaDB · FastAPI · Ollama · BM25 · RAGAS · Docker

---

## 版本三：技术深度版（适合算法岗，展示技术理解）

### 项目经历

**领克汽车智能问答系统 — 从检索到评估的完整 RAG 实现** | 独立开发 | *2025.06 - 2025.07*

**背景**：针对 LLM 在垂直领域存在的幻觉和知识时效性问题，设计并实现了一套完整的 RAG 系统，解决中文汽车产品知识库的精准检索与可信生成问题。

**核心工作**：

1. **文档处理层**
   - 使用 PyMuPDF 解析 PDF 产品手册，实现中文文本清洗（去控制字符、规范化空白、去零宽字符）
   - 对比递归分块 / 语义分块 / 父子文档分块三种策略，通过实验选定 chunk_size=512 + overlap=128 的最优配置

2. **检索优化层**（核心创新）
   - 设计 Dense（ChromaDB + qwen3-embedding）+ Sparse（BM25 + jieba 中文分词）混合检索，权重配比 0.7:0.3
   - 引入 Cross-Encoder Reranker（BGE-reranker-large）实现"粗召回→精排序"两阶段检索
   - 实现 Multi-Query Rewriting：利用 LLM 从多角度改写用户问题，多个等价查询并行检索后去重合并，Recall@5 提升至 0.82

3. **Agent 决策层**
   - 基于 LangChain Tool Calling Agent 实现自主工具调用（知识检索 / 贷款计算 / 车型对比）
   - 设计结构化 System Prompt 约束 Agent 行为边界，防止编造数据
   - 实现 Buffer / Window / Summary 三种记忆策略，解决长对话的 Token 预算管理

4. **评估与工程化**
   - 构建 RAGAS 四维评估体系（Faithfulness / Answer Relevancy / Context Recall / Context Precision），设计 5 组标注评估问题
   - 记录 5 组消融实验，量化每次策略调整的效果增量
   - FastAPI + SSE 流式输出，支持 Swagger 自动文档

**技术栈**：Python · LangChain · ChromaDB · FastAPI · Ollama · BM25 · BGE-Reranker · jieba · PyMuPDF · RAGAS

---

## 技术栈汇总（技能清单区域）

```markdown
编程语言   ：Python（熟练）、SQL（基础）
大模型相关 ：LangChain、RAG、Agent、Prompt Engineering、Function Calling
检索技术   ：向量检索（ChromaDB）、BM25、Cross-Encoder Reranker、混合检索
NLP工具    ：jieba、PyMuPDF、sentence-transformers
Web框架    ：FastAPI、SSE流式输出、RESTful API
评估框架   ：RAGAS（Faithfulness/Relevancy/Recall/Precision）
模型使用   ：通义千问（qwen）、Ollama 本地部署、BGE Embedding/Reranker
开发工具   ：Git、VS Code、Postman
```

---

## 面试时如何"讲"这个项目

### 30秒版本（自我介绍顺带提）

> "我最近独立完成了一个 RAG 智能问答项目，用 LangChain 搭了一套从文档解析到检索增强生成的完整 pipeline。核心创新是设计了三层检索策略——混合检索加重排序加查询重写，把召回率从比较低的水平提到了 0.82。还搭建了 RAGAS 评估体系来量化优化效果，最后用 FastAPI 做了服务化部署。"

### 3分钟版本（面试官追问"详细讲讲"）

> 按照这个结构讲：
>
> **1. 为什么做（30秒）**
> "LLM 有两个核心问题：幻觉和知识时效性。RAG 是目前最主流的解决方案——让模型基于外部知识库回答，而不是凭记忆。我选汽车领域是因为产品手册是现成的 PDF，场景清晰。"
>
> **2. 怎么做（90秒）—— 按检索 Pipeline 讲**
> "整个系统分四层：
> - 文档处理：PDF 解析 + 清洗 + 分块，我做了三种分块策略的对比实验
> - 检索层：这是核心。我发现单一检索有盲区——向量检索不理解"03"是车型代号，BM25 不理解"省油"和"油耗低"是一回事。所以做了混合检索，向量权重 0.7 + BM25 权重 0.3。然后加了 Cross-Encoder Reranker 做精排，最后用 Multi-Query 重写来覆盖用户不同的表达方式
> - Agent 层：挂载了三个工具——知识检索、贷款计算、车型对比，模型自己决定什么时候调用哪个
> - 服务层：FastAPI + SSE 流式输出"
>
> **3. 怎么验证效果（60秒）—— 用数据说话**
> "这是我觉得最关键的部分：我建了评估体系。标注了 5 个问题作为测试集，每次改策略都用 RAGAS 的四个指标重新评估。忠实度从 0.78 提到 0.88，Recall@5 从 0.60 提到 0.82。每个优化都有数据支撑，不是凭感觉调参。"

---

## 投递策略

| 公司类型 | 用哪个版本 | 强调什么 |
|----------|-----------|---------|
| 长安/赛力斯等车企 | 版本一 | 汽车领域知识、业务价值 |
| AI创业公司 | 版本三 | 技术深度、实验思维 |
| 互联网大厂 | 版本一或三 | 工程能力、完整度 |
| 传统企业AI部门 | 版本二 | 落地能力、ROI |