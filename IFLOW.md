# UAgent - OpenHands 增强研究系统

## 项目概述

**UAgent** 是基于 **OpenHands**（前身为 OpenDevin）的增强版本，是一个 AI 驱动的软件开发平台，通过 **UAgent 研究扩展** 添加了自主并行研究能力。该系统能够自动为复杂查询触发基于树的研究过程。

### 核心特性

- **自动研究触发**：智能识别复杂查询并自动启动研究模式
- **并行探索**：同时探索多种解决方案路径
- **PUCT 算法**：使用 AlphaZero 风格的树搜索算法平衡探索与利用
- **实时可视化**：通过 Web UI 实时查看研究树生长过程
- **预算控制**：自动管理成本和迭代次数
- **多适配器架构**：支持网络研究、代码分析和代码执行

## 项目架构

### 目录结构

```
/Users/wuy/Desktop/code/UAgent/
├── OpenHands/                          # 修改后的 OpenHands 主代码库
│   ├── openhands/                      # 核心 OpenHands 平台
│   │   ├── server/                     # FastAPI 后端服务器
│   │   ├── agenthub/                   # 代理实现
│   │   ├── controller/                 # 代理控制逻辑
│   │   ├── runtime/                    # 执行环境（Docker 等）
│   │   ├── llm/                        # LLM 集成层
│   │   └── events/                     # 事件系统
│   ├── extensions/                     # 扩展系统
│   │   └── uagent_research/            # ⭐ UAgent 研究扩展
│   │       ├── classifier/             # 任务分类（触发研究）
│   │       ├── middleware/             # 请求拦截
│   │       ├── orchestrator/           # 树搜索编排器（PUCT 算法）
│   │       ├── adapters/               # 执行适配器
│   │       ├── router/                 # 技能路由
│   │       ├── tools/                  # 研究工具
│   │       └── api/                    # 研究 API 端点
│   └── frontend/                       # React 前端
├── original_openhands/                 # 原始 OpenHands（参考）
├── workspace/                          # 用户工作空间
└── requirements.txt                    # Python 依赖
```

### 核心组件

#### 1. 任务分类器 (Task Classifier)
- **位置**: `extensions/uagent_research/classifier/task_classifier.py`
- **功能**: 分析用户查询，判断是否应触发研究模式
- **触发条件**: 
  - 复杂度指标 >= 0.7（可配置）
  - 包含研究关键词
  - 多阶段任务

#### 2. 研究中间件 (Research Middleware)
- **位置**: `extensions/uagent_research/middleware/research_middleware.py`
- **功能**: 拦截用户消息，在适当时触发研究
- **集成点**: 
  - 首条消息: `openhands/server/services/conversation_service.py`
  - 后续消息: `openhands/server/session/session.py`

#### 3. 树搜索编排器 (Tree Orchestrator)
- **位置**: `extensions/uagent_research/orchestrator/tree_orchestrator.py`
- **功能**: 使用 PUCT 算法执行核心研究引擎
- **PUCT 公式**: `PUCT(node) = Q(node) + c * P(node) * sqrt(N(parent)) / (1 + N(node))`

#### 4. 适配器 (Adapters)
- **DeepResearchAdapter**: 网络研究和浏览
- **RepoMasterAdapter**: 代码仓库分析
- **CodeActAdapter**: 代码执行

## 构建和运行

### 环境准备

1. **安装依赖**
```bash
cd /Users/wuy/Desktop/code/UAgent
pip install -r requirements.txt
```

2. **配置环境变量**
```bash
# 复制配置文件
cp .env.example .env

# 编辑配置，添加 API 密钥
vim .env
```

3. **关键配置项**
```bash
# 启用自动研究触发
ENABLE_AUTO_RESEARCH_TRIGGER=true

# 研究置信度阈值
RESEARCH_CONFIDENCE_THRESHOLD=0.7

# 预算限制
RESEARCH_MAX_ITERATIONS=50
RESEARCH_MAX_COST=10.0
RESEARCH_MAX_PARALLEL=3

# LLM 配置
LLM_API_KEY=your_api_key_here
LLM_MODEL=dashscope/qwen3-coder-plus
```

### 启动服务

```bash
# 使用提供的启动脚本
cd /Users/wuy/Desktop/code/UAgent
./start_openhands_research.sh

# 或手动启动
cd OpenHands
source ../.venv/bin/activate
python -m openhands.server
```

### 访问界面

- **主界面**: http://localhost:3000
- **研究 API**: http://localhost:3000/api/research
- **WebSocket**: ws://localhost:3000/api/research/ws

## 开发约定

### 代码风格

- 使用 **Python 3.9+** 语法
- 遵循 **PEP 8** 代码规范
- 所有公共函数必须有 **类型提示**
- 所有类和公共方法必须有 **文档字符串**

### 测试规范

```bash
# 运行所有测试
cd OpenHands/extensions/uagent_research
pytest -v

# 运行特定测试
pytest tests/test_models.py -v

# 生成覆盖率报告
pytest --cov=. --cov-report=html
```

### 添加新适配器

1. 创建适配器目录
2. 实现 `BaseAdapter` 接口
3. 在 `adapters/__init__.py` 中注册
4. 更新 `router/skill_router.py` 路由逻辑
5. 添加测试

### 调试技巧

```bash
# 测试分类器
python -c "
from extensions.uagent_research.classifier.task_classifier import task_classifier
result = task_classifier.should_trigger_research('your query')
print(f'Trigger: {result[0]}, Confidence: {result[2]:.2f}')
"

# 查看研究状态
curl http://localhost:3000/api/research/experiments/{experiment_id}/tree | jq

# 监控日志
tail -f logs/openhands.log | grep -i research
```

## API 接口

### 研究树状态
```bash
GET /api/research/experiments/{experiment_id}/tree
```

### 实验控制
```bash
# 暂停研究
POST /api/research/experiments/{experiment_id}/pause

# 恢复研究
POST /api/research/experiments/{experiment_id}/resume

# 停止研究
POST /api/research/experiments/{experiment_id}/stop
```

### 事件流
```bash
# 获取实验事件
GET /api/research/experiments/{experiment_id}/events?since_version=0&limit=100
```

## 核心流程

### 研究触发流程

1. **用户提交查询**
2. **任务分类器分析** → 计算置信度分数
3. **中间件拦截** → 如果置信度 >= 阈值，触发研究
4. **创建编排器** → 初始化研究树
5. **后台运行** → 非阻塞执行研究
6. **返回用户** → "研究模式已激活"

### PUCT 树搜索

```
循环直到预算耗尽：
1. SELECT: 使用 PUCT 选择最佳节点
2. EXPAND: 生成子节点
3. EXECUTE: 并行执行子节点
4. BACKPROPAGATE: 更新节点值
5. PUBLISH: 发布树状态到 API
```

### 节点类型

- **ROOT**: 用户原始查询
- **IDEA**: 高级方法
- **HYPOTHESIS**: 具体策略
- **EXPERIMENT**: 具体测试/实现

### 节点状态

- **PENDING**: 未开始
- **RUNNING**: 进行中
- **COMPLETE**: 成功完成
- **FAILED**: 执行失败

## 故障排除

### 研究未触发

1. 检查配置：`grep ENABLE_AUTO_RESEARCH_TRIGGER extensions/uagent_research/config.py`
2. 重启服务器：配置更改后必须重启
3. 测试复杂度：使用已知复杂查询测试

### 前端卡住

1. 检查 API：`curl http://localhost:3000/api/research/experiments`
2. 查看日志：`tail -f logs/openhands.log`
3. 禁用研究：`export ENABLE_AUTO_RESEARCH_TRIGGER=false`

### 成本过高

1. 降低预算：`export RESEARCH_MAX_COST=5.0`
2. 减少迭代：`export RESEARCH_MAX_ITERATIONS=20`
3. 提高阈值：`export RESEARCH_CONFIDENCE_THRESHOLD=0.8`

## 性能优化

### 成本控制

```bash
# 使用更便宜的模型
export RESEARCH_DEFAULT_MODEL=gpt-4o-mini

# 减少并行分支
export RESEARCH_MAX_PARALLEL=2

# 缩短迭代
export RESEARCH_MAX_ITERATIONS=30
```

### 性能监控

- **节点/迭代**: 2-5（正常）
- **成本/节点**: $0.05-$0.50（典型）
- **成功率**: 70-90%（良好）
- **深度**: 2-4 层（最优）

## 扩展开发

### 自定义适配器

```python
# adapters/my_adapter/adapter.py
from ..base.adapter import BaseAdapter

class MyAdapter(BaseAdapter):
    async def run(self, task, context):
        # 实现逻辑
        yield Event(...)
```

### 修改 PUCT 参数

```python
# orchestrator/tree_orchestrator.py
EXPLORATION_CONSTANT = 1.414  # 默认值（sqrt(2)）
# 更高 = 更多探索
# 更低 = 更多利用
```

### 添加工具

1. 在 `tools/` 创建工具
2. 在 `tools/__init__.py` 注册
3. 在适配器中使用

## 文档资源

### 核心文档

- **项目概述**: `CODEBASE_OVERVIEW.md`
- **机制说明**: `UAGENT_MECHANISM_EXPLAINED.md`
- **快速参考**: `QUICK_REFERENCE.md`
- **架构图**: `ARCHITECTURE_DIAGRAM.md`
- **创新点**: `INNOVATIONS_AND_DIFFERENCES.md`

### 关键文件

- **任务分类器**: `extensions/uagent_research/classifier/task_classifier.py`
- **树编排器**: `extensions/uagent_research/orchestrator/tree_orchestrator.py`
- **研究中间件**: `extensions/uagent_research/middleware/research_middleware.py`
- **研究 API**: `extensions/uagent_research/api/research_routes.py`

## 许可证

MIT License

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 编写测试
4. 提交 Pull Request

---

*IFLOW.md v1.0 - 最后更新: 2025-01-06*