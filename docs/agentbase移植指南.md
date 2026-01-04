# AgentBase 迁移到 DeepAgents 指南

## 概述

本文档指导如何将 AgentBase 的优秀组件迁移到 DeepAgents 架构中，实现"骨架 + 灵魂"的最佳组合。

**核心思路**：DeepAgents 提供了一个很好的骨架（图构建、Middleware 架构、SubAgent 系统），而 AgentBase 拥有更好的血肉（AU2 压缩算法、工具并发管理）。将 AgentBase 组件移植到 DeepAgents 架构中，是性价比最高的选择。

---

## 迁移清单

| AgentBase 组件 | 迁移方式 | 优先级 | 状态 |
|---------------|---------|--------|------|
| `AU2Compressor` | → `AU2CompressionMiddleware` | ⭐⭐⭐ 高 | 待迁移 |
| `ExtendedToolRegistry` 并发锁 | → `ConcurrentSafeToolMiddleware` | ⭐⭐⭐ 高 | 待迁移 |
| 权限检查 `check_permissions` | → `PermissionMiddleware.wrap_tool_call` | ⭐⭐ 中 | 待迁移 |
| 超时控制 `timeout` | → `TimeoutMiddleware.awrap_tool_call` | ⭐⭐ 中 | 待迁移 |
| 用户确认 `requires_consent` | 已有 `HumanInTheLoopMiddleware` | ✅ 已覆盖 | 无需迁移 |

---

## 核心组件迁移

### 1. AU2 压缩器 → AU2CompressionMiddleware

#### 背景对比

| 特性 | DeepAgents SummarizationMiddleware | AgentBase AU2Compressor |
|------|-----------------------------------|------------------------|
| 压缩结构 | 简单文本摘要 | **8 段式结构化**（会话元信息、核心目标、关键决策等） |
| 触发机制 | token 数量/比例 | **倒序扫描 + 80% 阈值** |
| 系统提示词保护 | ❌ 可能被覆盖 | ✅ **自动识别并保留** |
| 最近消息保留 | 基于数量/比例 | **可配置 keep_recent_messages** |
| 预留输出空间 | ❌ 不考虑 | ✅ **预留 max_output_tokens** |

#### 迁移实现

```python
# middleware/au2_compression.py

from typing import Any
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime
from langgraph.types import Overwrite

from agentbase.memory.compressor import AU2Compressor


class AU2CompressionMiddleware(AgentMiddleware):
    """AU2 8段式压缩 Middleware
    
    替换 DeepAgents 默认的 SummarizationMiddleware，提供更精细的上下文压缩。
    
    特性：
    - 8段式结构化摘要（会话元信息、核心目标、关键决策等）
    - 倒序扫描获取 token 使用量
    - 80% 阈值触发压缩
    - 自动保护原始 SystemMessage
    - 预留输出 token 空间
    """
    
    def __init__(
        self,
        model=None,
        token_trigger_ratio: float = 0.80,
        max_output_tokens: int = 8192,
        keep_recent_messages: int = 5,
        context_limit: int = 128000,
    ):
        """初始化 AU2 压缩 Middleware
        
        Args:
            model: 用于压缩的 LLM 模型
            token_trigger_ratio: 触发压缩的阈值（默认 80%）
            max_output_tokens: 预留的输出 token 数（默认 8192）
            keep_recent_messages: 保留的最近消息数（默认 5）
            context_limit: 上下文限制（默认 128000）
        """
        super().__init__()
        self.compressor = AU2Compressor(
            llm_provider=model,
            token_trigger_ratio=token_trigger_ratio,
            max_output_tokens=max_output_tokens,
            keep_recent_messages=keep_recent_messages,
        )
        self.context_limit = context_limit
    
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """在模型调用前检查并执行压缩"""
        messages = state.get("messages", [])
        
        if not self.compressor.should_compress(messages, self.context_limit):
            return None
        
        # 执行压缩（同步版本）
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果在异步上下文中，创建新任务
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, 
                        self.compressor.compress(messages)
                    )
                    result = future.result()
            else:
                result = asyncio.run(self.compressor.compress(messages))
        except RuntimeError:
            result = asyncio.run(self.compressor.compress(messages))
        
        # 重建消息列表
        new_messages = []
        if result.original_system_message:
            new_messages.append(result.original_system_message)
        new_messages.append(result.compressed_message)
        new_messages.extend(result.messages_to_keep)
        
        return {"messages": Overwrite(new_messages)}
```

#### 使用方式

```python
from deepagents import create_deep_agent
from middleware.au2_compression import AU2CompressionMiddleware

# 注意：需要排除默认的 SummarizationMiddleware
agent = create_deep_agent(
    model=model,
    tools=[...],
    middleware=[
        AU2CompressionMiddleware(
            model=model,
            token_trigger_ratio=0.80,
            context_limit=128000,
        ),
        # 其他 middleware...
    ],
)
```

---

### 2. 工具并发锁 → ConcurrentSafeToolMiddleware

#### 实现方式

DeepAgents 的 `AgentMiddleware` 提供了 `wrap_tool_call` 钩子，可以在工具执行时触发：

```python
# middleware/concurrent_safe.py

import asyncio
from typing import Any, Callable, Awaitable
from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from agentbase.tools.extended_registry import ExtendedToolRegistry, get_extended_registry


class ConcurrentSafeToolMiddleware(AgentMiddleware):
    """并发安全工具执行 Middleware
    
    为非并发安全的工具自动加锁，防止并发冲突。
    
    使用方式：
    1. 在 ExtendedToolRegistry 中注册工具时标记 concurrent_safe=False
    2. 添加此 Middleware 到 agent
    3. 自动为非并发安全的工具加锁
    """
    
    def __init__(self, registry: ExtendedToolRegistry | None = None):
        """初始化并发安全 Middleware
        
        Args:
            registry: 扩展工具注册表，不传则使用全局实例
        """
        super().__init__()
        self.registry = registry or get_extended_registry()
        self._locks: dict[str, asyncio.Lock] = {}
    
    def _get_lock(self, tool_name: str) -> asyncio.Lock:
        """获取或创建工具锁"""
        if tool_name not in self._locks:
            self._locks[tool_name] = asyncio.Lock()
        return self._locks[tool_name]
    
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """异步包装工具调用，为非并发安全的工具加锁"""
        tool_name = request.call.name
        
        # 检查是否注册在扩展注册表中
        if not self.registry.has(tool_name):
            return await handler(request)
        
        meta = self.registry.get_meta(tool_name)
        
        if not meta.concurrent_safe:
            # 非并发安全的工具需要加锁
            async with self._get_lock(tool_name):
                return await handler(request)
        
        # 并发安全的工具直接执行
        return await handler(request)
    
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """同步版本（回退到无锁执行）"""
        # 同步版本无法使用 asyncio.Lock，直接执行
        return handler(request)
```

#### 使用方式

```python
from langchain_core.tools import tool
from agentbase.tools.extended_registry import get_extended_registry, ExtendedToolMeta
from middleware.concurrent_safe import ConcurrentSafeToolMiddleware

# 1. 注册工具并标记并发安全性
registry = get_extended_registry()

@tool
def write_to_database(data: str) -> str:
    """写入数据库（非并发安全）"""
    # ...

registry.register(
    write_to_database,
    ExtendedToolMeta(concurrent_safe=False)  # 标记为非并发安全
)

# 2. 创建 agent 时添加 Middleware
agent = create_deep_agent(
    model=model,
    tools=registry.get_tools(),
    middleware=[
        ConcurrentSafeToolMiddleware(registry),
        # 其他 middleware...
    ],
)
```

---

### 3. 工具超时控制 → TimeoutMiddleware

```python
# middleware/timeout.py

import asyncio
from typing import Any, Callable, Awaitable
from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from agentbase.tools.extended_registry import ExtendedToolRegistry, get_extended_registry


class ToolTimeoutMiddleware(AgentMiddleware):
    """工具超时控制 Middleware"""
    
    def __init__(
        self, 
        registry: ExtendedToolRegistry | None = None,
        default_timeout: float = 30.0,
    ):
        super().__init__()
        self.registry = registry or get_extended_registry()
        self.default_timeout = default_timeout
    
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """异步包装工具调用，添加超时控制"""
        tool_name = request.call.name
        
        # 获取超时设置
        if self.registry.has(tool_name):
            timeout = self.registry.get_timeout(tool_name)
        else:
            timeout = self.default_timeout
        
        try:
            return await asyncio.wait_for(handler(request), timeout=timeout)
        except asyncio.TimeoutError:
            return ToolMessage(
                content=f"工具 {tool_name} 执行超时（{timeout}秒）",
                name=tool_name,
                tool_call_id=request.call.id,
            )
```

---

### 4. 权限检查 → PermissionMiddleware

```python
# middleware/permissions.py

from typing import Any, Callable, Awaitable
from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from agentbase.tools.extended_registry import ExtendedToolRegistry, get_extended_registry


class PermissionMiddleware(AgentMiddleware):
    """工具权限检查 Middleware"""
    
    def __init__(
        self, 
        registry: ExtendedToolRegistry | None = None,
        user_permissions: list[str] | None = None,
    ):
        super().__init__()
        self.registry = registry or get_extended_registry()
        self.user_permissions = user_permissions or []
    
    def set_user_permissions(self, permissions: list[str]) -> None:
        """动态设置用户权限"""
        self.user_permissions = permissions
    
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """检查权限后执行工具"""
        tool_name = request.call.name
        
        if self.registry.has(tool_name):
            if not self.registry.check_permissions(tool_name, self.user_permissions):
                return ToolMessage(
                    content=f"权限不足：无法执行工具 {tool_name}",
                    name=tool_name,
                    tool_call_id=request.call.id,
                )
        
        return await handler(request)
```

---

## 推荐的目录结构

```
deepagents-extensions/
├── __init__.py
├── middleware/
│   ├── __init__.py
│   ├── au2_compression.py      # AU2 8段式压缩
│   ├── concurrent_safe.py      # 并发安全控制
│   ├── timeout.py              # 超时控制
│   └── permissions.py          # 权限检查
├── tools/
│   ├── __init__.py
│   └── extended_registry.py    # 扩展工具注册表（从 AgentBase 迁移）
└── prompt/
    ├── __init__.py
    └── au2.py                  # AU2 压缩提示词
```

---

## 完整使用示例

```python
"""使用迁移后的 Middleware 创建增强版 DeepAgent"""

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

# 导入迁移后的 Middleware
from deepagents_extensions.middleware import (
    AU2CompressionMiddleware,
    ConcurrentSafeToolMiddleware,
    ToolTimeoutMiddleware,
    PermissionMiddleware,
)
from deepagents_extensions.tools import ExtendedToolRegistry, ExtendedToolMeta

# 1. 初始化模型
model = init_chat_model(
    model="openai:deepseek-chat",
    api_key="your-api-key",
    base_url="https://api.deepseek.com",
)

# 2. 定义工具
@tool
def search_web(query: str) -> str:
    """搜索网络"""
    return f"搜索结果: {query}"

@tool  
def write_file(path: str, content: str) -> str:
    """写入文件（非并发安全）"""
    return f"已写入 {path}"

# 3. 注册工具到扩展注册表
registry = ExtendedToolRegistry()
registry.register(search_web, ExtendedToolMeta(
    concurrent_safe=True,
    timeout=60.0,
))
registry.register(write_file, ExtendedToolMeta(
    concurrent_safe=False,  # 写文件不能并发
    timeout=30.0,
    permissions=["file:write"],
))

# 4. 创建增强版 Agent
agent = create_deep_agent(
    model=model,
    tools=registry.get_tools(),
    system_prompt="你是一个功能强大的 AI 助手。",
    checkpointer=MemorySaver(),
    backend=FilesystemBackend(root_dir="./workspace"),
    middleware=[
        # 替换默认的摘要中间件
        AU2CompressionMiddleware(
            model=model,
            token_trigger_ratio=0.80,
            context_limit=128000,
        ),
        # 并发安全控制
        ConcurrentSafeToolMiddleware(registry),
        # 超时控制
        ToolTimeoutMiddleware(registry, default_timeout=30.0),
        # 权限检查
        PermissionMiddleware(registry, user_permissions=["file:write"]),
    ],
)

# 5. 运行
result = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
```

---

## 迁移路径

```
Phase 1: 快速验证 (1-2天)
├── 直接使用 DeepAgents，不做任何修改
├── 验证基本功能正常
├── 熟悉 stream_mode="debug" 的日志输出
└── 理解 Middleware 执行流程

Phase 2: 注入 AU2 压缩 (2-3天)
├── 迁移 AU2Compressor 代码
├── 编写 AU2CompressionMiddleware
├── 替换默认的 SummarizationMiddleware
└── 测试长对话场景的压缩效果

Phase 3: 增加并发控制 (1-2天)
├── 迁移 ExtendedToolRegistry
├── 编写 ConcurrentSafeToolMiddleware
├── 添加 TimeoutMiddleware
└── 验证并发场景

Phase 4: 打包复用 (1天)
├── 创建 deepagents-extensions 包
├── 编写文档和测试
└── 团队共享使用
```

---

## 注意事项

### 1. 避免 Fork 魔改

DeepAgents 会持续更新，如果 fork 一份魔改版，以后合并上游更新会很痛苦。**尽量通过 Middleware 扩展，而不是修改核心代码**。

### 2. Middleware 执行顺序

- `before_*` 钩子按添加顺序执行
- `after_*` 钩子按逆序执行
- `wrap_*` 钩子形成洋葱模型

### 3. 调试技巧

```python
# 查看图结构
print(agent.get_graph().draw_mermaid())

# 详细日志
async for chunk in agent.astream(
    {"messages": [...]},
    stream_mode="debug",
    subgraphs=True,
):
    print(chunk)
```

---

## 参考资料

- [DeepAgents 源码](https://github.com/langchain-ai/deepagents)
- [LangGraph Middleware 文档](https://langchain-ai.github.io/langgraph/)
- [AgentBase AU2 压缩算法](../agentbase/agentbase/memory/compressor.py)
- [AgentBase 扩展工具注册表](../agentbase/agentbase/tools/extended_registry.py)
