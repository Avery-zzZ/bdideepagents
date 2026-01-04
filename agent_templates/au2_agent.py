"""Deepagents with customized middlewares - 使用自定义中间件替代默认配置。

当前自定义中间件:
1. AU2CompressionMiddleware - 8段式结构化压缩, 更适合代码开发场景
   - 保留技术细节和代码上下文
   - 倒序扫描获取真实 token 使用量
   - 预留输出 token 空间

使用方式:
    from langchain.chat_models import init_chat_model
    from deepagents.graph_customized import create_deep_agent_customized

    model = init_chat_model("openai:deepseek-chat", base_url="...", api_key="...")
    agent = create_deep_agent_customized(
        model=model,
        tools=[...],
        max_context_window=128000,  # DeepSeek 输入上限
        max_output_tokens=8192,     # DeepSeek 输出上限
    )
"""

from collections.abc import Callable, Sequence
from typing import Any

from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import (
    CompiledSubAgent,
    SubAgent,
    SubAgentMiddleware,
)
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    TodoListMiddleware,
)
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain.agents.structured_output import ResponseFormat
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from middlewares.au2_compression import AU2CompressionMiddleware, ContextSize

BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."

# 默认值
DEFAULT_CONTEXT_WINDOW = 131072  # 默认 128K
DEFAULT_MAX_OUTPUT_TOKENS = 8192  # 默认 8K


def _get_context_window(model: BaseChatModel, max_context_window: int | None) -> int:
    """获取模型的上下文窗口大小。

    优先级:
    1. 用户显式设置的 max_context_window
    2. 模型的 profile.max_input_tokens
    3. 使用默认值 DEFAULT_CONTEXT_WINDOW
    """
    if max_context_window is not None:
        return max_context_window

    # 尝试从模型 profile 获取
    if hasattr(model, "profile") and isinstance(model.profile, dict) and "max_input_tokens" in model.profile:
        return model.profile["max_input_tokens"]

    return DEFAULT_CONTEXT_WINDOW


def _get_max_output_tokens(model: BaseChatModel, max_output_tokens: int | None) -> int:
    """获取模型的最大输出 token 数。"""
    if max_output_tokens is not None:
        return max_output_tokens

    # 尝试从模型 profile 获取
    if hasattr(model, "profile") and isinstance(model.profile, dict) and "max_output_tokens" in model.profile:
        return model.profile["max_output_tokens"]

    return DEFAULT_MAX_OUTPUT_TOKENS


def create_au2_deep_agent(  # noqa: PLR0913
    model: BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    response_format: ResponseFormat | None = None,
    state_schema: type[AgentState[Any]] | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
    # 自定义中间件参数
    au2_triggers: list[ContextSize] | ContextSize | None = ("fraction", 0.8),
    au2_keep: ContextSize | None = ("messages", 5),
    model_context_length: int | None = None,
    model_max_output: int | None = None,
) -> CompiledStateGraph:
    r"""Create a deep agent with customized middlewares.

    使用自定义中间件配置, 当前使用 AU2 8段式结构化压缩, 更适合代码开发场景。

    Args:
        model: 必需, 使用的模型实例(通过 init_chat_model 创建)。
        tools: The tools the agent should have access to.
        system_prompt: The additional instructions the agent should have.
        middleware: Additional middleware to apply after standard middleware.
        subagents: The subagents to use.
        response_format: A structured output response format to use for the agent.
        context_schema: The schema of the deep agent.
        checkpointer: Optional checkpointer for persisting agent state between runs.
        store: Optional store for persistent storage.
        backend: Optional backend for file storage and execution.
        interrupt_on: Optional Dict[str, bool | InterruptOnConfig] mapping tool names to interrupt configs.
        debug: Whether to enable debug mode.
        name: The name of the agent.
        cache: The cache to use for the agent.

        # 压缩参数
        au2_triggers: 触发压缩条件，支持同时存在多个(各自触发)。默认("fraction", 0.8)。
            ("fraction", f): 上下文长度到达模型上限\*f时触发。
            ("tokens", n): 上下文长度到达n时触发。
            ("messages", n): 总消息数到达n时触发。
        au2_keep: 压缩后保留的原上下文。默认("messages", 5)。
            ("fraction", f): 保留原上下文长度为模型上限\*f。
            ("tokens", n): 保留原上下文长度为n。
            ("messages", n): 保留n条消息。
        model_context_length: 模型的最大上下文窗口大小(tokens)。
            如果不设置, 默认使用 128K。请根据实际模型设置。
        model_max_output: 预留的最大输出 token 数。触发压缩的阈值(0-1之间的比例)。默认 0.80 表示 80% 时触发。
            如果不设置, 默认使用 8K。请根据实际模型设置。


    Returns:
        A configured deep agent with customized middlewares.

    Examples:
        # 使用 DeepSeek
        model = init_chat_model("openai:deepseek-chat", base_url="...", api_key="...")
        agent = create_deep_agent_customized(
            model=model,
            tools=[...],
            max_context_window=128000,  # DeepSeek 输入上限
            max_output_tokens=8192,     # DeepSeek 输出上限
        )

        # 自定义压缩参数
        agent = create_deep_agent_customized(
            model=my_model,
            tools=[...],
            max_context_window=200000,  # Claude 3 输入上限
            max_output_tokens=4096,     # Claude 3 输出上限
            compression_trigger=0.85,   # 85% 时触发
            messages_to_keep=10,        # 保留最近 10 条
        )
    """
    # 获取上下文窗口大小
    context_window = _get_context_window(model, model_context_length)
    output_tokens = _get_max_output_tokens(model, model_max_output)


    # 创建 AU2 压缩中间件
    def create_au2_middleware() -> AU2CompressionMiddleware:
        return AU2CompressionMiddleware(
            model=model,
            trigger=au2_triggers,
            keep=au2_keep,
            max_context_window=context_window,
            max_output_tokens=output_tokens,
        )

    deepagent_middleware = [
        TodoListMiddleware(),
        FilesystemMiddleware(backend=backend),
        SubAgentMiddleware(
            default_model=model,
            default_tools=tools,
            subagents=subagents if subagents is not None else [],
            default_middleware=[
                TodoListMiddleware(),
                FilesystemMiddleware(backend=backend),
                create_au2_middleware(),  # 子 Agent 也使用 AU2
                AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
                PatchToolCallsMiddleware(),
            ],
            default_interrupt_on=interrupt_on,
            general_purpose_agent=True,
        ),
        create_au2_middleware(),  # 主 Agent 使用 AU2
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        PatchToolCallsMiddleware(),
    ]
    if middleware:
        deepagent_middleware.extend(middleware)
    if interrupt_on is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    return create_agent(
        model,
        system_prompt=system_prompt + "\n\n" + BASE_AGENT_PROMPT
        if system_prompt
        else BASE_AGENT_PROMPT,
        tools=tools,
        middleware=deepagent_middleware,
        response_format=response_format,
        state_schema=state_schema,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config({"recursion_limit": 1000})
