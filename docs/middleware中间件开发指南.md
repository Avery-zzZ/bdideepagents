# DeepAgents Middleware 开发指南

## 概述

Middleware（中间件）是 DeepAgents 的核心扩展机制，允许你在 Agent 执行的不同阶段注入自定义逻辑。通过 Middleware，你可以：

- 修改消息历史
- 注入自定义工具
- 扩展状态结构
- 拦截和修改模型请求/响应
- 拦截和修改工具调用

本文档将详细介绍如何实现不同类型的 Middleware。

---

## 设计哲学：AOP（面向切面编程）

DeepAgents 的 Middleware 架构采用了 **AOP（Aspect-Oriented Programming，面向切面编程）** 的设计思想。

### 什么是 AOP？

AOP 是一种编程范式，旨在通过将**横切关注点（Cross-Cutting Concerns）** 模块化来提高代码的模块性。横切关注点是那些跨越多个模块的功能，如：

- **日志记录** - 每个方法调用都需要记录
- **事务管理** - 多个数据库操作需要原子性
- **权限控制** - 每个敏感操作都需要检查权限
- **性能监控** - 需要测量各个组件的执行时间

AOP 的核心概念：

| 术语 | 含义 | DeepAgents 对应 |
|------|------|----------------|
| **Aspect（切面）** | 封装横切关注点的模块 | `AgentMiddleware` 类 |
| **Join Point（连接点）** | 程序执行中可以插入切面的点 | `before_model`、`wrap_tool_call` 等钩子 |
| **Advice（通知）** | 切面在连接点执行的代码 | 钩子方法的实现 |
| **Pointcut（切入点）** | 定义哪些连接点会被切面影响 | 通过钩子类型隐式定义 |
| **Weaving（织入）** | 将切面应用到目标对象的过程 | `create_deep_agent(middleware=[...])` |

### 传统 Agent vs Middleware Agent 对比

#### ❌ 传统方式：紧耦合的单体 Agent

```python
class TraditionalAgent:
    """传统 Agent - 所有功能耦合在一起"""
    
    def __init__(self, model, tools):
        self.model = model
        self.tools = tools
        self.message_history = []
        self.logger = Logger()
        self.permission_checker = PermissionChecker()
        self.metrics = MetricsCollector()
        
    def run(self, user_input):
        # 😰 日志记录散布在各处
        self.logger.info(f"Agent started with: {user_input}")
        
        # 😰 消息压缩逻辑混在主流程中
        if len(self.message_history) > 20:
            self.message_history = self._compress_messages()
        
        self.message_history.append({"role": "user", "content": user_input})
        
        while True:
            # 😰 权限检查代码重复
            if not self.permission_checker.check("model_call"):
                raise PermissionError("No permission to call model")
            
            # 😰 性能监控代码分散
            start_time = time.time()
            
            try:
                # 😰 重试逻辑嵌入主流程
                for attempt in range(3):
                    try:
                        response = self.model.invoke(self.message_history)
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise
                        time.sleep(1)
            finally:
                self.metrics.record("model_call", time.time() - start_time)
            
            self.message_history.append(response)
            
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    # 😰 工具权限检查重复
                    if not self.permission_checker.check(f"tool:{tool_call.name}"):
                        continue
                    
                    # 😰 超时控制混在工具执行中
                    try:
                        with timeout(30):
                            # 😰 并发控制逻辑分散
                            with self._get_lock(tool_call.name):
                                result = self._execute_tool(tool_call)
                    except TimeoutError:
                        result = "Tool execution timed out"
                    
                    self.message_history.append(result)
            else:
                break
        
        self.logger.info("Agent completed")
        return response.content
```

**传统方式的问题：**

| 问题 | 描述 |
|------|------|
| 🔴 **代码散乱（Scattering）** | 日志、权限检查等代码分散在各处 |
| 🔴 **代码缠绕（Tangling）** | 业务逻辑与横切关注点混在一起 |
| 🔴 **难以修改** | 修改日志策略需要改动多处代码 |
| 🔴 **难以复用** | 想在另一个 Agent 中复用权限检查？需要复制粘贴 |
| 🔴 **难以测试** | 无法单独测试压缩逻辑或重试逻辑 |
| 🔴 **难以理解** | 核心业务逻辑被大量辅助代码淹没 |

---

#### ✅ Middleware 方式：关注点分离

```python
# 📦 每个关注点独立封装为 Middleware

class LoggingMiddleware(AgentMiddleware):
    """日志记录 - 独立模块"""
    
    def before_agent(self, state, runtime):
        logging.info(f"Agent started: {state['messages'][-1]}")
    
    def after_agent(self, state, runtime):
        logging.info("Agent completed")


class CompressionMiddleware(AgentMiddleware):
    """消息压缩 - 独立模块"""
    
    def before_model(self, state, runtime):
        if len(state["messages"]) > 20:
            return {"messages": Overwrite(self._compress(state["messages"]))}


class RetryMiddleware(AgentMiddleware):
    """重试逻辑 - 独立模块"""
    
    async def awrap_model_call(self, request, handler):
        for attempt in range(3):
            try:
                return await handler(request)
            except Exception:
                if attempt == 2: raise
                await asyncio.sleep(1)


class MetricsMiddleware(AgentMiddleware):
    """性能监控 - 独立模块"""
    
    def wrap_model_call(self, request, handler):
        start = time.time()
        try:
            return handler(request)
        finally:
            metrics.record("model_call", time.time() - start)


class PermissionMiddleware(AgentMiddleware):
    """权限控制 - 独立模块"""
    
    async def awrap_tool_call(self, request, handler):
        if not self.check_permission(request.call.name):
            return ToolMessage("Permission denied", tool_call_id=request.call.id)
        return await handler(request)


class TimeoutMiddleware(AgentMiddleware):
    """超时控制 - 独立模块"""
    
    async def awrap_tool_call(self, request, handler):
        return await asyncio.wait_for(handler(request), timeout=30)


class ConcurrencyMiddleware(AgentMiddleware):
    """并发控制 - 独立模块"""
    
    async def awrap_tool_call(self, request, handler):
        async with self._get_lock(request.call.name):
            return await handler(request)


# ✨ 组合使用 - 清晰、灵活、可复用
agent = create_deep_agent(
    model=init_chat_model("openai:gpt-4o"),
    tools=[...],
    middleware=[
        LoggingMiddleware(),        # 日志
        CompressionMiddleware(),    # 压缩
        RetryMiddleware(),          # 重试
        MetricsMiddleware(),        # 监控
        PermissionMiddleware(),     # 权限
        TimeoutMiddleware(),        # 超时
        ConcurrencyMiddleware(),    # 并发
    ],
)
```

**Middleware 方式的优势：**

| 优势 | 描述 |
|------|------|
| 🟢 **关注点分离** | 每个 Middleware 只负责一个功能 |
| 🟢 **高度模块化** | 功能可以独立开发、测试、部署 |
| 🟢 **即插即用** | 添加/移除功能只需修改 middleware 列表 |
| 🟢 **易于复用** | 同一个 Middleware 可用于多个 Agent |
| 🟢 **易于测试** | 可以单独测试每个 Middleware |
| 🟢 **组合灵活** | 通过顺序控制执行优先级 |
| 🟢 **业务清晰** | Agent 核心逻辑不被横切关注点污染 |

---

### 洋葱模型（Onion Model）

`wrap_model_call` 和 `wrap_tool_call` 采用洋葱模型执行，类似于 Express.js、Koa 的中间件：

```
请求进入 ─────────────────────────────────────────────────> 响应返回
         │                                                   ↑
         ▼                                                   │
┌─────────────────────────────────────────────────────────────────────┐
│  Middleware A (最外层)                                              │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Middleware B                                                 │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  Middleware C (最内层)                                  │  │  │
│  │  │  ┌─────────────────────────────────────────────────┐    │  │  │
│  │  │  │                                                 │    │  │  │
│  │  │  │           实际 Model/Tool 调用                  │    │  │  │
│  │  │  │                                                 │    │  │  │
│  │  │  └─────────────────────────────────────────────────┘    │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

执行顺序（middleware=[A, B, C]）:
1. A 的前置逻辑
2. B 的前置逻辑
3. C 的前置逻辑
4. 实际调用
5. C 的后置逻辑
6. B 的后置逻辑
7. A 的后置逻辑
```

这种模式让每层 Middleware 都能：
- **拦截请求**：在调用 `handler()` 前修改请求
- **拦截响应**：在调用 `handler()` 后处理响应
- **短路返回**：不调用 `handler()` 直接返回结果
- **异常处理**：用 try/except 包裹 `handler()` 处理错误

---

## AgentMiddleware 核心接口

```python
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.runtime import Runtime
from typing import Any, Callable, Awaitable

class AgentMiddleware:
    """Middleware 基类 - 所有自定义 Middleware 都应继承此类"""
    
    # ============ 类属性 ============
    state_schema: type[AgentState] = AgentState  # 扩展状态结构
    tools: list[BaseTool] = []                    # 提供的工具列表
    
    # ============ 生命周期钩子 ============
    
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Agent 开始执行前调用（整个循环开始时）"""
        pass
    
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Agent 执行完成后调用（整个循环结束时）"""
        pass
    
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """模型调用前调用（每次 LLM 调用前）"""
        pass
    
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """模型调用后调用（每次 LLM 调用后）"""
        pass
    
    # ============ 包装式钩子 ============
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """包装模型调用 - 可修改请求和响应"""
        pass
    
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """异步版本的 wrap_model_call"""
        pass
    
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """包装工具调用 - 可修改工具执行前后的行为"""
        pass
    
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """异步版本的 wrap_tool_call"""
        pass
```

---

## 钩子执行顺序

```
Agent 执行流程:
┌─────────────────────────────────────────────────────────────────┐
│  before_agent  (按 Middleware 添加顺序执行)                      │
│       ↓                                                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  循环开始                                                   │ │
│  │       ↓                                                    │ │
│  │  before_model  (按顺序执行)                                 │ │
│  │       ↓                                                    │ │
│  │  wrap_model_call  (洋葱模型，外层先进后出)                   │ │
│  │       ↓                                                    │ │
│  │  [LLM 调用]                                                 │ │
│  │       ↓                                                    │ │
│  │  after_model  (按逆序执行)                                  │ │
│  │       ↓                                                    │ │
│  │  ┌───────────────────────────────────────────────────────┐ │ │
│  │  │  如果有工具调用:                                       │ │ │
│  │  │  wrap_tool_call  (洋葱模型)                           │ │ │
│  │  │       ↓                                              │ │ │
│  │  │  [工具执行]                                           │ │ │
│  │  └───────────────────────────────────────────────────────┘ │ │
│  │       ↓                                                    │ │
│  │  继续循环或结束                                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│       ↓                                                         │
│  after_agent  (按逆序执行)                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔑 核心钩子详解与对比

这一节详细对比 `before_agent`、`before_model`、`after_model`、`wrap_model_call` 等核心钩子的区别，帮助你选择正确的钩子。

### 一、生命周期钩子 vs 包装式钩子

Middleware 的钩子分为两大类：

| 类别 | 钩子 | 特点 |
|------|------|------|
| **生命周期钩子** | `before_agent`, `after_agent`, `before_model`, `after_model` | 在特定时机执行，**只能修改状态**，无法拦截/修改请求或响应 |
| **包装式钩子** | `wrap_model_call`, `wrap_tool_call` | 包裹实际调用，**可以拦截、修改、重试、短路** |

---

### 二、before_model vs wrap_model_call 详细对比

这是最常被混淆的两个钩子，它们都在模型调用前执行，但用途完全不同：

#### 执行时机对比

```
before_model 执行时机:
┌─────────────────────────────────────────────────────┐
│  before_model() ←── 在这里修改状态                   │
│       ↓                                             │
│  wrap_model_call() ←── 在这里包裹调用                │
│       ↓                                             │
│  [LLM 调用]                                          │
│       ↓                                             │
│  wrap_model_call() 后半部分 ←── 处理响应              │
│       ↓                                             │
│  after_model() ←── 在这里处理调用后的状态             │
└─────────────────────────────────────────────────────┘
```

#### 能力对比表

| 能力 | `before_model` | `wrap_model_call` |
|------|---------------|-------------------|
| **修改消息历史** | ✅ 可以 | ❌ 不可以（已封装在 request 中） |
| **修改系统提示词** | ❌ 不可以 | ✅ 可以（通过 `request.override(system_prompt=...)）` |
| **修改工具列表** | ❌ 不可以 | ✅ 可以（通过 `request.override(tools=...)）` |
| **替换模型** | ❌ 不可以 | ✅ 可以（通过 `request.override(model=...)）` |
| **拦截/短路调用** | ❌ 不可以 | ✅ 可以（不调用 `handler()` 直接返回） |
| **重试逻辑** | ❌ 不可以 | ✅ 可以（`try/except` 包裹 `handler()`） |
| **修改响应** | ❌ 不可以 | ✅ 可以（处理 `handler()` 返回值） |
| **异常处理** | ❌ 不可以 | ✅ 可以（`try/except` 包裹 `handler()`） |
| **性能计时** | ❌ 不可以 | ✅ 可以（在 `handler()` 前后计时） |
| **访问完整状态** | ✅ 可以（`state` 参数） | ✅ 可以（`request.state`） |
| **更新状态** | ✅ 可以（返回 dict） | ❌ 不可以（只能返回 ModelResponse） |

#### 场景选择指南

```
你的需求是什么？
       │
       ├─── 需要压缩/过滤消息历史？
       │         └── ✅ 使用 before_model
       │
       ├─── 需要在调用前更新自定义状态字段？
       │         └── ✅ 使用 before_model
       │
       ├─── 需要修改系统提示词？
       │         └── ✅ 使用 wrap_model_call
       │
       ├─── 需要动态替换工具列表？
       │         └── ✅ 使用 wrap_model_call
       │
       ├─── 需要实现重试逻辑？
       │         └── ✅ 使用 wrap_model_call
       │
       ├─── 需要缓存模型响应？
       │         └── ✅ 使用 wrap_model_call
       │
       ├─── 需要测量调用耗时？
       │         └── ✅ 使用 wrap_model_call
       │
       └─── 需要拦截并返回固定响应（如限流）？
                 └── ✅ 使用 wrap_model_call
```

#### 代码示例对比

**场景 1：消息压缩（使用 before_model）**

```python
class CompressionMiddleware(AgentMiddleware):
    """消息压缩 - 必须用 before_model"""
    
    def before_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        messages = state.get("messages", [])
        
        if len(messages) > 20:
            # 压缩消息
            compressed = self._compress(messages)
            return {"messages": Overwrite(compressed)}
        
        return None  # 不需要修改
```

**❌ 错误示范：用 wrap_model_call 做压缩**

```python
class WrongCompressionMiddleware(AgentMiddleware):
    """❌ 错误！wrap_model_call 无法修改消息历史"""
    
    def wrap_model_call(self, request, handler):
        # ❌ request.messages 是只读的，无法修改
        # ❌ 即使你在这里"修改"了，也不会影响实际调用
        # ❌ 因为 messages 已经在 before_model 之后被封装进 request 了
        return handler(request)
```

---

**场景 2：修改系统提示词（使用 wrap_model_call）**

```python
class SystemPromptMiddleware(AgentMiddleware):
    """注入系统提示词 - 必须用 wrap_model_call"""
    
    def wrap_model_call(self, request, handler):
        new_prompt = f"{request.system_prompt}\n\n你是一个专业的代码助手。"
        modified_request = request.override(system_prompt=new_prompt)
        return handler(modified_request)
```

**❌ 错误示范：用 before_model 改系统提示词**

```python
class WrongSystemPromptMiddleware(AgentMiddleware):
    """❌ 错误！before_model 无法修改系统提示词"""
    
    def before_model(self, state, runtime):
        # ❌ state 中没有 system_prompt 字段
        # ❌ 系统提示词是在 wrap_model_call 阶段才构建的
        # ❌ 即使你修改了 messages 中的 SystemMessage，也可能被覆盖
        return None
```

---

**场景 3：重试逻辑（使用 wrap_model_call）**

```python
class RetryMiddleware(AgentMiddleware):
    """重试逻辑 - 必须用 wrap_model_call"""
    
    async def awrap_model_call(self, request, handler):
        for attempt in range(3):
            try:
                return await handler(request)
            except Exception as e:
                if attempt == 2:
                    raise
                await asyncio.sleep(1)
```

**❌ 错误示范：用 before_model 做重试**

```python
class WrongRetryMiddleware(AgentMiddleware):
    """❌ 错误！before_model 在调用前执行，无法重试"""
    
    def before_model(self, state, runtime):
        # ❌ 这里还没调用 LLM，怎么重试？
        # ❌ before_model 只是修改状态的钩子
        return None
```

---

### 三、before_agent vs before_model 对比

| 维度 | `before_agent` | `before_model` |
|------|---------------|----------------|
| **执行次数** | 整个 Agent 运行期间**只执行一次** | **每次 LLM 调用前都执行** |
| **执行时机** | Agent 循环开始前 | 每次模型调用前（循环内） |
| **典型用途** | 初始化、修补悬空工具调用 | 消息压缩、Token 计数 |
| **适用场景** | 一次性操作 | 每轮都需要的操作 |

```
Agent 执行时间线:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  before_agent() ←── 只执行一次！                                 │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  第一轮循环                                              │    │
│  │  before_model() ←── 第 1 次执行                          │    │
│  │  wrap_model_call() → LLM → after_model()                │    │
│  │  [工具调用...]                                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  第二轮循环                                              │    │
│  │  before_model() ←── 第 2 次执行                          │    │
│  │  wrap_model_call() → LLM → after_model()                │    │
│  │  [工具调用...]                                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                         │
│       ▼                                                         │
│  ... 更多循环 ...                                                │
│       │                                                         │
│       ▼                                                         │
│  after_agent() ←── 只执行一次！                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 代码示例

**场景：修补悬空工具调用（使用 before_agent）**

```python
class PatchToolCallsMiddleware(AgentMiddleware):
    """修补悬空工具调用 - 使用 before_agent（只需执行一次）"""
    
    def before_agent(self, state, runtime):
        # 只在 Agent 开始时执行一次
        # 修补历史消息中的悬空工具调用
        patched = self._patch_dangling_tool_calls(state["messages"])
        return {"messages": Overwrite(patched)}
```

**场景：消息压缩（使用 before_model）**

```python
class CompressionMiddleware(AgentMiddleware):
    """消息压缩 - 使用 before_model（每轮都检查）"""
    
    def before_model(self, state, runtime):
        # 每次 LLM 调用前都检查是否需要压缩
        # 因为工具调用后消息会增加
        if self._should_compress(state["messages"]):
            compressed = self._compress(state["messages"])
            return {"messages": Overwrite(compressed)}
        return None
```

---

### 四、after_model 的用途

`after_model` 在模型调用**完成后**执行，可以：

- 记录模型响应到自定义状态
- 统计 Token 使用量
- 触发后续操作

```python
class TokenCounterMiddleware(AgentMiddleware):
    """Token 计数器"""
    
    state_schema = TokenCountState  # 扩展状态包含 total_tokens
    
    def after_model(self, state, runtime):
        # 模型响应已经在 state["messages"] 中
        last_message = state["messages"][-1]
        
        if hasattr(last_message, "usage_metadata"):
            tokens_used = last_message.usage_metadata.get("total_tokens", 0)
            current_total = state.get("total_tokens", 0)
            return {"total_tokens": current_total + tokens_used}
        
        return None
```

---

### 五、wrap_tool_call vs wrap_model_call 对比

两者都是包装式钩子，但针对不同的目标：

| 维度 | `wrap_model_call` | `wrap_tool_call` |
|------|-------------------|------------------|
| **包裹对象** | LLM 调用 | 工具执行 |
| **请求类型** | `ModelRequest` | `ToolCallRequest` |
| **返回类型** | `ModelResponse` | `ToolMessage` 或 `Command` |
| **典型用途** | 重试、缓存、系统提示词 | 超时、权限、并发控制 |
| **执行频率** | 每次 LLM 调用 | 每次工具调用（可能多个） |

---

### 六、钩子选择决策树

```
开始
 │
 ├─── 需要修改消息历史？
 │     ├── 只需要在开始时修改一次？ → before_agent
 │     └── 每轮都需要修改？ → before_model
 │
 ├─── 需要修改模型请求（提示词/工具/模型）？
 │     └── wrap_model_call
 │
 ├─── 需要处理模型调用的错误/重试/缓存？
 │     └── wrap_model_call
 │
 ├─── 需要在模型调用后更新状态？
 │     └── after_model
 │
 ├─── 需要控制工具执行（超时/权限/并发）？
 │     └── wrap_tool_call
 │
 └─── 需要在 Agent 结束时清理资源？
       └── after_agent
```

---

### 七、组合使用示例

一个完整的 Middleware 可能需要使用多个钩子：

```python
class AdvancedMiddleware(AgentMiddleware):
    """综合使用多个钩子的示例"""
    
    state_schema = MyExtendedState
    
    def before_agent(self, state, runtime):
        """初始化：设置起始时间"""
        return {"start_time": time.time()}
    
    def before_model(self, state, runtime):
        """每轮前：压缩消息"""
        if len(state["messages"]) > 50:
            return {"messages": Overwrite(self._compress(state["messages"]))}
        return None
    
    def wrap_model_call(self, request, handler):
        """包裹调用：添加系统提示 + 重试"""
        modified = request.override(
            system_prompt=f"{request.system_prompt}\n当前时间: {datetime.now()}"
        )
        
        for attempt in range(3):
            try:
                return handler(modified)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1)
    
    def after_model(self, state, runtime):
        """调用后：记录 Token"""
        # 统计 Token...
        return {"total_tokens": ...}
    
    async def awrap_tool_call(self, request, handler):
        """工具调用：超时控制"""
        try:
            return await asyncio.wait_for(handler(request), timeout=30)
        except asyncio.TimeoutError:
            return ToolMessage("Tool timed out", tool_call_id=request.call.id)
    
    def after_agent(self, state, runtime):
        """结束：计算总耗时"""
        elapsed = time.time() - state.get("start_time", 0)
        print(f"Agent 运行耗时: {elapsed:.2f}s")
        return None
```

---

## Middleware 类型分类

根据你要实现的功能，选择合适的钩子：

| 类型 | 使用的钩子 | 典型用例 |
|------|----------|---------|
| **状态扩展型** | `state_schema` | 添加自定义状态字段 |
| **工具注入型** | `tools` | 提供新工具给 Agent |
| **消息处理型** | `before_agent`, `before_model` | 压缩/过滤消息历史 |
| **模型拦截型** | `wrap_model_call` | 修改系统提示词、重试逻辑 |
| **工具拦截型** | `wrap_tool_call` | 并发控制、超时、权限检查 |
| **综合型** | 多个钩子组合 | 复杂的业务逻辑 |

---

## 类型一：状态扩展型 Middleware

**用途**：扩展 Agent 的状态结构，添加自定义字段。

### 示例：研究状态 Middleware

```python
from langchain.agents.middleware import AgentMiddleware, AgentState


class ResearchState(AgentState):
    """扩展状态，添加 research 字段"""
    research: str  # 存储研究结果


class ResearchMiddleware(AgentMiddleware):
    """只扩展状态，不提供工具"""
    state_schema = ResearchState
```

### 注意事项

1. **状态字段会自动合并**：多个 Middleware 的 `state_schema` 会被合并
2. **使用 Annotated 定义 reducer**：控制状态更新方式

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class MyState(AgentState):
    # messages 使用 add_messages reducer（追加而非覆盖）
    messages: Annotated[list, add_messages]
    
    # 自定义 reducer
    counter: Annotated[int, lambda old, new: old + new]
```

---

## 类型二：工具注入型 Middleware

**用途**：为 Agent 提供新的工具。

### 示例：天气工具 Middleware

```python
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool


@tool
def get_weather(location: str) -> str:
    """获取天气信息"""
    return f"The weather in {location} is sunny."


class WeatherMiddleware(AgentMiddleware):
    """提供天气查询工具"""
    tools = [get_weather]
```

### 示例：带状态注入的工具

```python
from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langgraph.types import Command
from langchain_core.messages import ToolMessage


@tool
def research_topic(topic: str, runtime: ToolRuntime) -> Command:
    """研究主题并保存到状态
    
    使用 ToolRuntime 可以：
    1. 访问当前状态: runtime.state
    2. 获取工具调用 ID: runtime.tool_call_id
    3. 返回 Command 更新状态
    """
    current_research = runtime.state.get("research", "")
    new_research = f"{current_research}\n\nResearching {topic}..."
    
    return Command(
        update={
            "research": new_research,  # 更新自定义状态
            "messages": [ToolMessage(new_research, tool_call_id=runtime.tool_call_id)],
        }
    )


class ResearchToolMiddleware(AgentMiddleware):
    state_schema = ResearchState  # 需要扩展状态
    tools = [research_topic]
```

### 注意事项

1. **工具自动注册**：`tools` 列表中的工具会自动添加到 Agent
2. **使用 ToolRuntime**：可以访问状态和返回 Command
3. **返回 Command vs 返回字符串**：
   - 返回字符串：简单响应
   - 返回 Command：需要更新状态时

---

## 类型三：消息处理型 Middleware

**用途**：在 Agent 执行前/后处理消息历史。

### 示例：消息压缩 Middleware

```python
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime
from langgraph.types import Overwrite
from typing import Any


class CompressionMiddleware(AgentMiddleware):
    """消息历史压缩 Middleware"""
    
    def __init__(self, max_messages: int = 20):
        super().__init__()
        self.max_messages = max_messages
    
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """在模型调用前压缩消息"""
        messages = state.get("messages", [])
        
        if len(messages) <= self.max_messages:
            return None  # 不需要压缩
        
        # 保留系统消息 + 最近 N 条消息
        system_messages = [m for m in messages if m.type == "system"]
        recent_messages = messages[-self.max_messages:]
        
        compressed = system_messages + recent_messages
        
        # 使用 Overwrite 完全替换消息列表
        return {"messages": Overwrite(compressed)}
```

### 示例：修补悬空工具调用

```python
class PatchToolCallsMiddleware(AgentMiddleware):
    """修补消息历史中的悬空工具调用"""
    
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """在 Agent 运行前，处理悬空的工具调用"""
        messages = state["messages"]
        if not messages:
            return None

        patched_messages = []
        for i, msg in enumerate(messages):
            patched_messages.append(msg)
            
            # 检查 AI 消息是否有未完成的工具调用
            if msg.type == "ai" and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    # 查找对应的工具响应
                    has_response = any(
                        m.type == "tool" and m.tool_call_id == tool_call["id"]
                        for m in messages[i:]
                    )
                    
                    if not has_response:
                        # 添加占位响应
                        patched_messages.append(
                            ToolMessage(
                                content=f"Tool call {tool_call['name']} was cancelled.",
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"],
                            )
                        )

        return {"messages": Overwrite(patched_messages)}
```

### 注意事项

1. **返回 None**：表示不修改状态
2. **使用 Overwrite**：完全替换状态值（而非合并）
3. **before_agent vs before_model**：
   - `before_agent`：整个 Agent 循环开始时，只执行一次
   - `before_model`：每次 LLM 调用前都会执行

---

## 类型四：模型拦截型 Middleware

**用途**：修改发送给模型的请求或处理模型的响应。

### 示例：系统提示词注入 Middleware

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from typing import Callable


class SystemPromptMiddleware(AgentMiddleware):
    """注入额外的系统提示词"""
    
    def __init__(self, additional_prompt: str):
        super().__init__()
        self.additional_prompt = additional_prompt
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """修改系统提示词"""
        new_prompt = (
            f"{request.system_prompt}\n\n{self.additional_prompt}"
            if request.system_prompt
            else self.additional_prompt
        )
        
        # 使用 override 创建修改后的请求
        modified_request = request.override(system_prompt=new_prompt)
        
        return handler(modified_request)
    
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """异步版本"""
        new_prompt = (
            f"{request.system_prompt}\n\n{self.additional_prompt}"
            if request.system_prompt
            else self.additional_prompt
        )
        return await handler(request.override(system_prompt=new_prompt))
```

### 示例：重试 Middleware

```python
import asyncio
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse


class RetryMiddleware(AgentMiddleware):
    """模型调用失败时自动重试"""
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        super().__init__()
        self.max_retries = max_retries
        self.delay = delay
    
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse:
        """带重试的模型调用"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await handler(request)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.delay * (attempt + 1))
        
        raise last_error
```

### ModelRequest 可修改的属性

```python
class ModelRequest:
    state: AgentState              # 当前状态（只读）
    messages: list[AnyMessage]     # 消息列表
    system_prompt: str | None      # 系统提示词
    system_message: SystemMessage  # 系统消息对象
    tools: list[BaseTool]          # 可用工具
    runtime: Runtime               # 运行时上下文
    
    def override(
        self,
        model: BaseChatModel = None,      # 替换模型
        tools: list = None,                # 替换工具列表
        system_prompt: str = None,         # 替换系统提示词
        system_message: SystemMessage = None,  # 替换系统消息
    ) -> ModelRequest:
        """创建修改后的请求副本"""
        pass
```

---

## 类型五：工具拦截型 Middleware

**用途**：拦截和修改工具调用的行为。

### 示例：并发安全 Middleware

```python
import asyncio
from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from typing import Callable, Awaitable


class ConcurrentSafeMiddleware(AgentMiddleware):
    """为非并发安全的工具加锁"""
    
    def __init__(self, non_concurrent_tools: list[str]):
        super().__init__()
        self.non_concurrent_tools = set(non_concurrent_tools)
        self._locks: dict[str, asyncio.Lock] = {}
    
    def _get_lock(self, tool_name: str) -> asyncio.Lock:
        if tool_name not in self._locks:
            self._locks[tool_name] = asyncio.Lock()
        return self._locks[tool_name]
    
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """异步工具调用包装"""
        tool_name = request.call.name
        
        if tool_name in self.non_concurrent_tools:
            async with self._get_lock(tool_name):
                return await handler(request)
        
        return await handler(request)
```

### 示例：超时控制 Middleware

```python
import asyncio
from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest


class TimeoutMiddleware(AgentMiddleware):
    """工具执行超时控制"""
    
    def __init__(self, default_timeout: float = 30.0):
        super().__init__()
        self.default_timeout = default_timeout
    
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        """带超时的工具调用"""
        try:
            return await asyncio.wait_for(
                handler(request),
                timeout=self.default_timeout,
            )
        except asyncio.TimeoutError:
            return ToolMessage(
                content=f"Tool {request.call.name} timed out after {self.default_timeout}s",
                name=request.call.name,
                tool_call_id=request.call.id,
            )
```

### 示例：权限检查 Middleware

```python
class PermissionMiddleware(AgentMiddleware):
    """工具调用权限检查"""
    
    def __init__(self, restricted_tools: dict[str, list[str]]):
        """
        Args:
            restricted_tools: {工具名: [所需权限列表]}
        """
        super().__init__()
        self.restricted_tools = restricted_tools
        self.user_permissions: list[str] = []
    
    def set_permissions(self, permissions: list[str]):
        """设置当前用户权限"""
        self.user_permissions = permissions
    
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        """检查权限后执行"""
        tool_name = request.call.name
        
        if tool_name in self.restricted_tools:
            required = self.restricted_tools[tool_name]
            if not all(p in self.user_permissions for p in required):
                return ToolMessage(
                    content=f"Permission denied: {tool_name} requires {required}",
                    name=tool_name,
                    tool_call_id=request.call.id,
                )
        
        return await handler(request)
```

### ToolCallRequest 属性

```python
class ToolCallRequest:
    call: ToolCall        # 工具调用信息
    tool_call: dict       # 工具调用字典 {"name": ..., "args": ..., "id": ...}
    runtime: ToolRuntime  # 运行时，包含 state 和 tool_call_id
```

---

## 类型六：综合型 Middleware（完整示例）

### 示例：文件系统 Middleware（参考 DeepAgents 实现）

```python
from typing import Annotated, Any, Callable, Awaitable, NotRequired
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from typing_extensions import TypedDict


# 1. 定义文件数据结构
class FileData(TypedDict):
    content: list[str]
    created_at: str
    modified_at: str


# 2. 定义 reducer（控制状态更新方式）
def file_reducer(left: dict | None, right: dict) -> dict:
    """文件状态合并，支持删除（None 值）"""
    if left is None:
        return {k: v for k, v in right.items() if v is not None}
    result = {**left}
    for key, value in right.items():
        if value is None:
            result.pop(key, None)  # 删除
        else:
            result[key] = value    # 更新
    return result


# 3. 扩展状态
class FilesystemState(AgentState):
    files: Annotated[NotRequired[dict[str, FileData]], file_reducer]


# 4. 定义工具
@tool
def read_file(path: str) -> str:
    """读取文件内容"""
    # 实现...
    return f"Content of {path}"


@tool
def write_file(path: str, content: str) -> str:
    """写入文件"""
    # 实现...
    return f"Written to {path}"


# 5. 实现 Middleware
class FilesystemMiddleware(AgentMiddleware):
    """文件系统 Middleware - 综合示例"""
    
    # 扩展状态
    state_schema = FilesystemState
    
    # 提供工具
    tools = [read_file, write_file]
    
    def __init__(self, max_file_size: int = 100000):
        super().__init__()
        self.max_file_size = max_file_size
    
    # 修改系统提示词
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """注入文件系统使用说明"""
        fs_prompt = """## Filesystem Tools
You have access to read_file and write_file tools.
All paths must be absolute (start with /)."""
        
        new_prompt = (
            f"{request.system_prompt}\n\n{fs_prompt}"
            if request.system_prompt
            else fs_prompt
        )
        return handler(request.override(system_prompt=new_prompt))
    
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse:
        """异步版本"""
        # 同上...
        return await handler(request)
    
    # 拦截大文件
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        """拦截过大的工具结果"""
        result = handler(request)
        
        if isinstance(result, ToolMessage) and len(result.content) > self.max_file_size:
            # 截断过长的内容
            truncated = result.content[:self.max_file_size] + "\n... (truncated)"
            return ToolMessage(
                content=truncated,
                tool_call_id=result.tool_call_id,
            )
        
        return result
    
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        """异步版本"""
        result = await handler(request)
        # 同上处理...
        return result
```

---

## 常见错误和注意事项

### 1. 同步 vs 异步

```python
# ❌ 错误：在同步方法中使用 await
def wrap_model_call(self, request, handler):
    return await handler(request)  # 错误！

# ✅ 正确：同步方法调用同步 handler
def wrap_model_call(self, request, handler):
    return handler(request)

# ✅ 正确：异步方法使用 await
async def awrap_model_call(self, request, handler):
    return await handler(request)
```

### 2. 返回值

```python
# before_* / after_* 钩子：
# - 返回 None：不修改状态
# - 返回 dict：合并到状态
# - 返回 {"messages": Overwrite([...])}：替换状态

# wrap_* 钩子：
# - 必须返回结果（不能返回 None）
# - 必须调用 handler 或返回替代结果
```

### 3. Overwrite 的使用

```python
from langgraph.types import Overwrite

# 合并更新（默认行为）
return {"messages": new_messages}  # 追加到现有消息

# 完全替换
return {"messages": Overwrite(new_messages)}  # 替换整个消息列表
```

### 4. 钩子执行顺序

```python
# Middleware 列表：[A, B, C]

# before_* 钩子：A → B → C（顺序执行）
# after_* 钩子：C → B → A（逆序执行）
# wrap_* 钩子：洋葱模型 A(B(C(handler)))
```

### 5. 工具的 ToolRuntime

```python
@tool
def my_tool(param: str, runtime: ToolRuntime):
    """使用 runtime 访问状态"""
    # 访问状态
    current_value = runtime.state.get("my_field")
    
    # 获取工具调用 ID（返回 Command 时需要）
    tool_call_id = runtime.tool_call_id
    
    # 返回 Command 更新状态
    return Command(
        update={
            "my_field": new_value,
            "messages": [ToolMessage("Done", tool_call_id=tool_call_id)],
        }
    )
```

---

## 使用示例

```python
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

# 创建 Agent 并添加多个 Middleware
agent = create_deep_agent(
    model=init_chat_model("openai:gpt-4o"),
    tools=[],
    middleware=[
        # 消息压缩（最先执行）
        CompressionMiddleware(max_messages=20),
        
        # 系统提示词注入
        SystemPromptMiddleware("Always be helpful and concise."),
        
        # 工具超时控制
        TimeoutMiddleware(default_timeout=30.0),
        
        # 并发安全控制
        ConcurrentSafeMiddleware(non_concurrent_tools=["write_file"]),
        
        # 权限检查（最后执行工具拦截）
        PermissionMiddleware(restricted_tools={"delete_file": ["admin"]}),
    ],
)

# 运行
result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})
```

---

## 快速参考表

| 需求 | 使用的属性/方法 | 示例 |
|------|----------------|------|
| 添加新工具 | `tools = [...]` | WeatherMiddleware |
| 扩展状态 | `state_schema = MyState` | ResearchMiddleware |
| 修改消息历史 | `before_agent()` / `before_model()` | CompressionMiddleware |
| 修改系统提示词 | `wrap_model_call()` + `request.override()` | SystemPromptMiddleware |
| 修改工具列表 | `wrap_model_call()` + `request.override(tools=...)` | - |
| 工具执行前拦截 | `wrap_tool_call()` / `awrap_tool_call()` | PermissionMiddleware |
| 工具结果处理 | `wrap_tool_call()` 处理返回值 | FilesystemMiddleware |
| 更新状态 | 返回 `{"field": value}` | - |
| 替换状态 | 返回 `{"field": Overwrite(value)}` | PatchToolCallsMiddleware |
| 终止执行 | 返回 `{"jump_to": "end"}` | - |

---

## Middleware 架构的优缺点分析

### ✅ 优点

| 优点 | 描述 |
|------|------|
| **关注点分离** | 每个 Middleware 专注于单一功能，代码清晰易维护 |
| **高度可复用** | 同一个 Middleware 可在不同 Agent 间共享使用 |
| **即插即用** | 添加/移除功能只需修改 middleware 列表，无需改动核心代码 |
| **易于测试** | 可以独立单元测试每个 Middleware |
| **组合灵活** | 通过顺序控制执行优先级，动态组合功能 |
| **低耦合** | Middleware 之间通过标准接口通信，互不依赖 |
| **渐进式迁移** | 可以逐步将功能迁移到 Middleware，无需一次性重构 |
| **标准化扩展** | LangChain 官方接口，生态兼容性好 |

### ❌ 缺点与限制

| 缺点 | 描述 | 应对方案 |
|------|------|----------|
| **学习曲线** | 需要理解 AOP 概念和钩子执行顺序 | 阅读本文档 + 实践 |
| **调试复杂** | 多层 Middleware 嵌套时调试困难 | 添加日志 Middleware、使用 LangSmith |
| **性能开销** | 每个钩子都会增加一定开销 | 合并相关 Middleware、避免不必要的 wrap |
| **隐式行为** | 行为被 Middleware 修改但代码中不可见 | 良好的命名和文档 |
| **顺序敏感** | Middleware 顺序影响行为，易出错 | 明确文档化顺序要求 |
| **状态传递受限** | Middleware 间共享数据需通过 state | 合理设计 state_schema |

### ⚠️ Middleware 无法完成的场景

以下场景**必须使用 LangGraph 原生图修改**：

| 场景 | 原因 | 解决方案 |
|------|------|----------|
| **自定义图结构** | Middleware 无法添加/删除节点或边 | 使用 `StateGraph` 自定义 |
| **条件分支路由** | Middleware 无法控制流程走向（除了简单的跳转到 end） | 使用 `add_conditional_edges` |
| **并行节点执行** | Middleware 是顺序执行的 | 使用图的 `fan-out` 模式 |
| **子图嵌套** | 需要复杂的图嵌套结构 | 使用 `StateGraph` + `subgraph` |
| **人机交互中断** | 需要在特定节点暂停等待输入 | 使用 `interrupt_before/after` |
| **检查点恢复** | 需要从特定节点恢复执行 | 使用 LangGraph 的 checkpointing |
| **多 Agent 协作** | 复杂的多 Agent 通信模式 | 自定义图结构或 SubAgentMiddleware |

---

## 从 AgentBase 迁移到 Middleware 的原因

### 为什么要迁移？

AgentBase 使用的是**自定义图结构**方式，而 DeepAgents 使用 **Middleware + 标准图**方式：

```
AgentBase 方式（自定义图）:
┌─────────────────────────────────────────────────────┐
│  自定义 StateGraph                                  │
│  ┌─────┐    ┌─────────┐    ┌─────────┐    ┌─────┐  │
│  │Start│───>│Compress │───>│  Model  │───>│Tools│  │
│  └─────┘    └─────────┘    └─────────┘    └─────┘  │
│                  ↑                            │     │
│                  └────────────────────────────┘     │
└─────────────────────────────────────────────────────┘
问题：每次添加功能都要修改图结构

DeepAgents 方式（Middleware + 标准图）:
┌─────────────────────────────────────────────────────┐
│  标准 Agent 图（固定）                               │
│  ┌─────┐    ┌─────────┐    ┌─────────┐             │
│  │Start│───>│  Model  │───>│  Tools  │──> End     │
│  └─────┘    └─────────┘    └─────────┘             │
│       ↑          ↑              ↑                   │
│       │          │              │                   │
│  ┌────┴──────────┴──────────────┴────┐             │
│  │         Middleware 层              │             │
│  │  [Compress, Retry, Permission...] │             │
│  └───────────────────────────────────┘             │
└─────────────────────────────────────────────────────┘
优势：功能通过 Middleware 注入，图结构保持稳定
```

### 迁移收益对比

| 方面 | AgentBase（自定义图） | DeepAgents（Middleware） |
|------|----------------------|-------------------------|
| **添加新功能** | 修改图结构，可能影响现有流程 | 添加 Middleware，不影响现有代码 |
| **功能复用** | 需要复制图节点代码 | Middleware 直接复用 |
| **测试** | 需要测试整个图 | 可以独立测试每个 Middleware |
| **维护成本** | 图结构复杂时难以维护 | Middleware 列表清晰明了 |
| **团队协作** | 多人修改图结构易冲突 | 各自开发 Middleware，合并时只改列表 |
| **升级兼容** | LangGraph 升级可能需要大量修改 | Middleware 接口稳定，升级平滑 |

### 何时选择哪种方式？

```
决策树：
┌─────────────────────────────────────────────────────┐
│  你的需求是什么？                                    │
└───────────────────────────┬─────────────────────────┘
                            ↓
          ┌─────────────────┴─────────────────┐
          ↓                                   ↓
   ┌──────────────┐                    ┌──────────────┐
   │ 横切关注点    │                    │ 流程结构修改  │
   │ (日志/权限/  │                    │ (分支/并行/  │
   │  压缩/监控)  │                    │  子图/中断)  │
   └──────┬───────┘                    └──────┬───────┘
          ↓                                   ↓
   ┌──────────────┐                    ┌──────────────┐
   │ 使用         │                    │ 使用         │
   │ Middleware   │                    │ LangGraph 图 │
   └──────────────┘                    └──────────────┘
```

### 推荐的混合策略

对于复杂项目，需要理解 **`create_deep_agent` 的局限性**：

```python
# 查看 create_deep_agent 源码：
def create_deep_agent(...) -> CompiledStateGraph:
    # ...
    deepagent_middleware = [
        TodoListMiddleware(),
        FilesystemMiddleware(backend=backend),
        SubAgentMiddleware(...),
        SummarizationMiddleware(...),
        # ...
    ]
    
    # 最终调用的是 create_agent - 一个固定的 ReAct 图！
    return create_agent(
        model,
        system_prompt=...,
        tools=tools,
        middleware=deepagent_middleware,  # 只能通过 middleware 扩展
        # 没有 graph 参数！无法传入自定义图！
    ).with_config({"recursion_limit": 1000})
```

**关键事实**：
- `create_deep_agent` 和 `create_agent` 都是**固定的 ReAct 循环图**
- **没有 `graph` 参数** - 你无法传入自定义图结构
- 只能通过 `middleware` 参数来扩展功能

#### 场景一：适合使用 DeepAgents + Middleware

```python
# ✅ ReAct 模式：LLM 自主决定调用什么工具，循环直到完成
#    这正是 create_deep_agent 的设计目的

agent = create_deep_agent(
    model=model,
    tools=[search, write_file, execute],  # 工具由 LLM 自主选择
    middleware=[
        LoggingMiddleware(),
        CompressionMiddleware(),
        PermissionMiddleware(),
    ],
)

# 用户只给一个目标，LLM 自己规划执行步骤
result = agent.invoke({"messages": [
    {"role": "user", "content": "帮我写一个 Python 爬虫并测试它"}
]})
```

#### 场景二：不适合使用 DeepAgents，必须自定义图

```python
# ❌ 固定工作流：每一步的输入输出都是确定的
#    这种情况下 ReAct Agent 是错误的选择！

# 例如：文档处理流水线
# Step 1: 解析 PDF → Step 2: 提取表格 → Step 3: 翻译 → Step 4: 生成报告
# 每一步都是固定的，不需要 LLM 来"决定"下一步做什么

from langgraph.graph import StateGraph, START, END

class DocumentState(TypedDict):
    pdf_path: str
    parsed_text: str
    tables: list[dict]
    translated: str
    report: str

def build_document_pipeline():
    graph = StateGraph(DocumentState)
    
    # 固定的线性流程
    graph.add_node("parse", parse_pdf_node)
    graph.add_node("extract", extract_tables_node)
    graph.add_node("translate", translate_node)
    graph.add_node("report", generate_report_node)
    
    graph.add_edge(START, "parse")
    graph.add_edge("parse", "extract")
    graph.add_edge("extract", "translate")
    graph.add_edge("translate", "report")
    graph.add_edge("report", END)
    
    return graph.compile()

# 这种工作流不需要 Middleware，因为：
# 1. 没有"横切关注点" - 每个节点就是业务逻辑
# 2. 没有 ReAct 循环 - 不需要 LLM 做决策
# 3. 结构固定 - 就是一条流水线
```

#### 场景三：需要条件分支的工作流

```python
# ❌ 条件分支工作流：根据条件走不同路径
#    create_deep_agent 无法实现这种结构

from langgraph.graph import StateGraph, START, END

def build_customer_service():
    graph = StateGraph(ServiceState)
    
    graph.add_node("classify", classify_intent_node)
    graph.add_node("refund", handle_refund_node)
    graph.add_node("technical", handle_technical_node)
    graph.add_node("sales", handle_sales_node)
    
    graph.add_edge(START, "classify")
    
    # 条件分支 - Middleware 无法做到
    graph.add_conditional_edges("classify", route_by_intent, {
        "refund": "refund",
        "technical": "technical", 
        "sales": "sales",
    })
    
    graph.add_edge("refund", END)
    graph.add_edge("technical", END)
    graph.add_edge("sales", END)
    
    return graph.compile()
```

#### 场景四：在自定义图中使用 Middleware 思想

如果你需要自定义图，但仍想要 Middleware 的好处，可以**手动组合**：

```python
from langgraph.graph import StateGraph, START, END

def build_hybrid_agent():
    graph = StateGraph(MyState)
    
    # 自定义入口节点
    graph.add_node("preprocess", preprocess_node)
    
    # 在中间嵌入一个 ReAct Agent（作为子图）
    react_agent = create_deep_agent(
        model=model,
        tools=tools,
        middleware=[CompressionMiddleware(), PermissionMiddleware()],
    )
    graph.add_node("agent", react_agent)
    
    # 自定义出口节点
    graph.add_node("postprocess", postprocess_node)
    
    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "agent")
    graph.add_edge("agent", "postprocess")
    graph.add_edge("postprocess", END)
    
    return graph.compile()
```
```

---

## 总结

### DeepAgents (create_deep_agent) 的本质

```
create_deep_agent = create_agent + 预配置 Middleware
                  = 固定的 ReAct 图 + Middleware 扩展

┌─────────────────────────────────────────────────────────┐
│  ReAct 循环（固定结构，不可修改）                        │
│                                                         │
│   START ──> Model ──> Tools ──> Model ──> ... ──> END  │
│              ↑          │                               │
│              └──────────┘                               │
│                                                         │
│   Middleware 在这个循环的各个点进行拦截和扩展            │
└─────────────────────────────────────────────────────────┘
```

### 适用场景对比

| 场景类型 | 使用什么 | 原因 |
|---------|---------|------|
| **自主决策 Agent** | `create_deep_agent` + Middleware | LLM 自己决定调用什么工具，ReAct 循环 |
| **固定流水线** | 自定义 `StateGraph` | 每步固定，不需要 LLM 决策 |
| **条件分支工作流** | 自定义 `StateGraph` | 需要 `add_conditional_edges` |
| **并行处理** | 自定义 `StateGraph` | 需要 `fan-out` 模式 |
| **人机交互** | 自定义 `StateGraph` + `interrupt` | 需要在特定节点暂停 |
| **混合模式** | 自定义图 + 嵌入 ReAct Agent | 大框架自定义，内部用 Agent |

### 何时使用 Middleware vs 自定义图

| 维度 | 使用 Middleware | 使用自定义 LangGraph 图 |
|------|----------------|------------------------|
| **决策者** | LLM 决定下一步 | 代码/条件决定下一步 |
| **流程结构** | ReAct 循环（固定） | 任意图结构（灵活） |
| **工具调用** | LLM 选择调用哪个工具 | 代码指定调用哪个节点 |
| **适用任务** | 开放式任务（"帮我完成X"） | 结构化任务（流水线/工作流） |
| **扩展方式** | 添加 Middleware | 添加/修改节点和边 |

### 最佳实践

1. 🤔 **先问自己**：任务是需要 LLM 自主决策，还是流程已经固定？
   - 自主决策 → `create_deep_agent` + Middleware
   - 流程固定 → 自定义 `StateGraph`

2. 🎯 **Middleware 用于横切关注点**：
   - ✅ 日志、监控、权限、压缩、重试、缓存
   - ❌ 业务流程控制、条件分支、并行执行

3. 🔧 **自定义图用于流程控制**：
   - ✅ 固定流水线、条件路由、并行处理、人机交互
   - ❌ 简单的 ReAct Agent（用 create_deep_agent 更简单）

4. 🧩 **可以混合使用**：
   - 自定义图作为外层框架
   - 在某个节点嵌入 `create_deep_agent` 作为子图

---
