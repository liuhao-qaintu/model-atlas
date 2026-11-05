# Model Atlas · 深度学习模型知识库

Model Atlas 将模型学习、文献阅读与跨资料问答连接在同一工作空间。围绕原理、输入输出、公式与代码建立模型知识，通过可定位的引用、文献分析报告和有依据的模型关系，支持连续追问与比较研究。

## 应用场景

研究人员阅读故障诊断论文时，发现作者使用了自编码器。可以先查看 AE 的输入、潜在表示和重构目标，再导入论文，阅读其方法与创新点报告，比较论文方法与普通 AE 的差异。追问时保留当前会话和已选文献，资料不足时按需补查其他获准处理的知识，并通过引用核对结论。

## 核心能力

| 入口 | 用途 |
| --- | --- |
| 学模型 | 阅读原理、输入输出、公式、代码解释及登记的论文图示，围绕当前章节继续提问 |
| 读文献 | 导入 PDF、Markdown 或文本；后台切分，生成整篇分析报告，确认标签与关系 |
| 比较研究 | 拆解比较问题，按证据缺口多轮查找，整理带引用的比较结果并继续问答 |
| 智能问答 | 结合会话确定指代与问题范围，检索相关正文，流式返回回答与引用 |
| 模型关系 | 探索模型、模块、方法与论文之间有原文依据的联系 |
| 知识管理 | 管理自定义模型、文献版本、云端处理许可与检索优先级 |

维护人员另可使用关系审核、检索对照与回答评测工具。普通用户页面优先展示答案、资料和引用，调试信息按权限隔离。

## 实现方式

- **前端**：Vue 3、Vue Router、Vite、JavaScript；Markdown、公式与代码高亮使用已有渲染组件。
- **后端**：FastAPI、SQLAlchemy、MySQL，SSE 返回生成进度与正文。
- **检索**：BM25、向量、实体/章节召回，经 RRF 融合与重排；子片段命中后补齐父章节。
- **GraphRAG**：沿有原文摘录支持的模型关系补查相关资料；采用关系记录与本地遍历。
- **Agentic RAG**：LangGraph 编排问题分解、检索、证据核对与有界补查；资料研究最多三轮查找。
- **评测**：检索指标与逐项回答标准分开记录，机器建议与人工复核分开统计。

知识匹配、关系扩展和会话存储在服务端执行；嵌入、重排和生成调用配置的兼容模型 API。文献需获得云端处理许可，才会用于这些远程操作。

## 安装与启动

需要 Python 3.12、MySQL 8 和支持当前 Vite 版本的 Node.js。以下 PowerShell 示例使用项目内的 `.venv` 虚拟环境；使用其他环境时替换解释器路径。

保持模型卡资料目录与项目同级：

```text
ModelAtlas/
  model-rag/
  模型知识库资料/
```

在 `model-rag` 根目录安装后端依赖，为前端安装锁定依赖：

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
npm.cmd --prefix frontend ci
```

仅在尚无 `.env` 时复制 `.env.example`，填写 `MYSQL_*`、模型服务配置和 `AUTH_SECRET_KEY`。签名密钥至少 32 个字符，可使用虚拟环境的 Python 生成随机值：

```powershell
& '.\.venv\Scripts\python.exe' -c "import secrets; print(secrets.token_urlsafe(48))"
& '.\.venv\Scripts\python.exe' -m backend.infra.bootstrap
& '.\.venv\Scripts\python.exe' tools/manage_accounts.py create-admin --username atlas_admin
```

创建管理员时在终端交互输入两遍密码，密码不作为命令参数。普通账号在注册页创建。

分别在两个终端启动后端与前端：

```powershell
# 项目根目录
& '.\.venv\Scripts\python.exe' -m uvicorn backend.main:app --host 127.0.0.1 --port 8002 --reload --reload-dir backend
```

```powershell
# 项目根目录
npm.cmd --prefix frontend run dev
```

访问 [Model Atlas](http://127.0.0.1:5174/)，使用已创建的账号登录。在对应终端按 Ctrl+C 停止服务。启动不依赖 WeKnora 或 Docker。

## 可选原始资料

初次启动可直接阅读内置模型卡，并上传自己的资料。论文原文件与原论文图片不随源码分发；未安装图示资料时展示文字及来源入口。

从项目根目录运行以下命令，可下载登记的固定公开来源，并以自己的账号导入。导入时终端要求登录，文件及检索数据归当前账号；这些公开资料按允许云端处理入库。

```powershell
& '.\.venv\Scripts\python.exe' tools/import_public_sources.py --download-only
& '.\.venv\Scripts\python.exe' tools/import_public_sources.py --import-only
```

导入完毕后可在“读文献”中准备检索索引。官方接口资料可通过`tools/prepare_official_interfaces.py`的同名参数下载和导入。下载和导入不执行远程源码；没有运行这些命令时，不会自动下载论文或请求云端模型。

## 账号与资料归属

预置模型卡对登录用户共享；个人文献、自建模型与关系、分析报告、会话及向量缓存按账号隔离。业务管理员也不能直接查看其他用户的私有资料。

“我的文献”只列出当前账号上传或导入的资料，可按标题、文件名和标签查找。公共参考来源通过模型内容和回答引用查看，不再单独占用资料目录入口。

维护功能同时要求 `admin` 角色和本地 `MAINTAINER_USER_IDS` 白名单。创建账号后，将终端输出的账号 ID 配入 `.env`，重启后端后生效：

```dotenv
MAINTAINER_USER_IDS=["管理员账号ID"]
```

已有单用户数据升级后保留原内容，未归属的私有数据默认不可见。确认归属后，可在首次创建管理员时显式接管：

```powershell
& '.\.venv\Scripts\python.exe' tools/manage_accounts.py create-admin --username atlas_admin --claim-legacy
```

该参数只分配尚无所有者的非公共旧数据，不改变已有归属。已创建同名账号时命令会拒绝覆盖；本地账号维护支持 `list` 与 `reset-password`。

认证使用 HttpOnly Cookie，写操作校验 CSRF。修改密码后原登录凭证失效。HTTPS 部署时设置 `AUTH_COOKIE_SECURE=true`；接口文档默认关闭，受控开发环境可通过 `ENABLE_API_DOCS=true` 开启。

## 代码导航

| 路径 | 职责 |
| --- | --- |
| `backend/main.py` | 应用启动、问答、流式接口 |
| `backend/infra/auth.py`、`access.py` | 认证、请求身份与数据库归属约束 |
| `backend/infra/secure_retriever.py` | 按账号建立检索视图 |
| `backend/rag/` | 查询规划、召回、图扩展与 Agentic RAG |
| `backend/generation/` | 证据组织、生成约束、引用校验与评估 |
| `backend/api/` | 认证、文献、研究、会话与维护接口 |
| `frontend/src/views/` | 模型学习、文献、问答与账号页面 |
| `tools/manage_accounts.py` | 本地管理员初始化与账号维护 |
| `tests/test_auth_access.py` | 真实认证链路与多账号隔离回归 |

## 数据与验证范围

PDF 解析结果用于检索和整篇分析，不还原论文排版；扫描件需先进行 OCR。引用提供文件、章节、页码及解析段落定位，公式和实验结论应结合原文核对。系统不执行论文模型训练，也不以功能测试数量代替回答准确率。

向量以 NumPy 缓存，不依赖独立向量数据库。任务队列位于单进程内，进程重启后运行中任务需重新发起；已经保存的文献、会话和报告仍由数据库保留。

仓库配置模板不包含实际数据库密码或模型密钥；运行库、个人上传资料、会话、向量缓存及内部开发记录不属于发布内容。论文与代码资料保留各自来源，知识条目区分原始资料、整理说明与教学示例。

项目设计参考 WeKnora 的知识库交互方式与 all-in-rag 的检索编排思路。
