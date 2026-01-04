"""AU2 上下文压缩中间件 - 智能对话历史压缩算法。

本模块实现了 AU2（Adaptive Unified Ultracompression）压缩算法作为 DeepAgents Middleware，
用于在 LLM 上下文窗口接近限制时，智能压缩对话历史，同时保留关键信息。

核心特性
========
1. **倒序扫描优化** - 从最新消息开始查找 token 使用信息，时间复杂度 O(k)
2. **智能触发阈值** - 默认 80% 触发压缩（可配置）
3. **预留输出空间** - 计算可用 token 时预留 max_output_tokens
4. **8段式结构化压缩** - 生成结构化摘要，保留关键信息
5. **AI ↔ Tool 消息对保护** - 确保不会切断相关联的消息对
6. **系统提示词保护** - 自动识别并保留原始 SystemMessage

与 SummarizationMiddleware 的区别
================================
- AU2 使用 8 段式结构化摘要（更适合代码开发场景）
- AU2 采用倒序扫描获取真实 token 使用量（更精确）
- AU2 预留输出 token 空间（避免输出时超限）
- SummarizationMiddleware 使用自然语言摘要（更通用）

使用示例
========
    from deepagents import create_deep_agent
    from deepagents.middleware import AU2CompressionMiddleware
    
    agent = create_deep_agent(
        model=model,
        tools=tools,
        middleware=[
            AU2CompressionMiddleware(
                model=model,
                trigger=("fraction", 0.80),
                keep=("messages", 5),
            ),
        ],
    )
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    MessageLikeRepresentation,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

TokenCounter = Callable[[Iterable[MessageLikeRepresentation]], int]

# AU2 8段式压缩提示词
AU2_COMPRESSION_PROMPT = """
Your task is to create a detailed summary of the conversation so far, 
paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, 
and architectural decisions that would be essential for continuing development 
work without losing context.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing.

Respond ONLY with the structured summary. Do not include any additional commentary.

<messages>
{messages}
</messages>
"""

_DEFAULT_MESSAGES_TO_KEEP = 5
_DEFAULT_TRIM_TOKEN_LIMIT = 8192
_SEARCH_RANGE_FOR_TOOL_PAIRS = 5

ContextFraction = tuple[Literal["fraction"], float]
ContextTokens = tuple[Literal["tokens"], int]
ContextMessages = tuple[Literal["messages"], int]
ContextSize = ContextFraction | ContextTokens | ContextMessages


@dataclass(slots=True)
class CompressionResult:
    """压缩结果数据结构。
    
    Attributes:
        summary: 压缩后的摘要（8段式结构化摘要文本）
        token_saved: 节省的token数量
        compression_ratio: 压缩比率（0-1之间，越小压缩效果越好）
        original_tokens: 原始消息的 token 数量
        compressed_tokens: 压缩后的 token 数量
    """
    summary: str
    token_saved: int
    compression_ratio: float
    original_tokens: int = 0
    compressed_tokens: int = 0


class AU2CompressionMiddleware(AgentMiddleware):
    """AU2 智能上下文压缩中间件。
    
    采用 8 段式结构化压缩，特别适合代码开发场景。
    
    核心特性：
    1. 倒序扫描：从最新消息获取真实 token 使用量
    2. 智能触发：可配置的触发阈值
    3. 预留输出：避免 LLM 输出时超出上下文限制
    4. 结构化摘要：8 段式压缩，保留关键信息
    5. 消息对保护：确保 AI ↔ Tool 消息不被分离
    """

    # AU2 压缩摘要的标识前缀
    AU2_SUMMARY_PREFIX = "[AU2 Context Summary]"

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        trigger: ContextSize | list[ContextSize] | None = None,
        keep: ContextSize = ("messages", _DEFAULT_MESSAGES_TO_KEEP),
        max_context_window: int | None = None,
        max_output_tokens: int | None = None,
        token_counter: TokenCounter = count_tokens_approximately,
        compression_prompt: str = AU2_COMPRESSION_PROMPT,
        trim_tokens_to_summarize: int | None = _DEFAULT_TRIM_TOKEN_LIMIT,
    ) -> None:
        """初始化 AU2 压缩中间件。
        
        Args:
            model: 用于生成摘要的语言模型
            trigger: 触发压缩的阈值条件
                - ("fraction", 0.8): 达到模型最大输入的 80% 时触发
                - ("tokens", 100000): 达到 100000 tokens 时触发
                - ("messages", 50): 达到 50 条消息时触发
                - 可以传入列表，任一条件满足即触发
            keep: 压缩后保留的上下文
                - ("messages", 5): 保留最近 5 条消息
                - ("tokens", 3000): 保留最近 3000 tokens
                - ("fraction", 0.1): 保留 10% 的 token 容量
            max_context_window: 模型的最大上下文窗口大小（tokens）
                - 如果设置，将覆盖从模型 profile 获取的值
                - 使用 fraction 类型的 trigger/keep 时必须设置（除非模型有 profile）
                - 例如：DeepSeek 128K = 128000, GPT-4 128K = 128000
            max_output_tokens: 预留的最大输出 token 数
            token_counter: token 计数函数
            compression_prompt: 压缩提示词模板
            trim_tokens_to_summarize: 发送给压缩模型的最大 token 数
        """
        super().__init__()

        if isinstance(model, str):
            model = init_chat_model(model)

        self.model = model
        self.max_context_window = max_context_window
        self.max_output_tokens = max_output_tokens
        self.token_counter = token_counter
        self.compression_prompt = compression_prompt
        self.trim_tokens_to_summarize = trim_tokens_to_summarize

        # 处理触发条件
        if trigger is None:
            self.trigger: ContextSize | list[ContextSize] | None = None
            trigger_conditions: list[ContextSize] = []
        elif isinstance(trigger, list):
            validated_list = [self._validate_context_size(item, "trigger") for item in trigger]
            self.trigger = validated_list
            trigger_conditions = validated_list
        else:
            validated = self._validate_context_size(trigger, "trigger")
            self.trigger = validated
            trigger_conditions = [validated]
        self._trigger_conditions = trigger_conditions

        self.keep = self._validate_context_size(keep, "keep")

        # 检查是否需要模型 profile 信息
        requires_profile = any(condition[0] == "fraction" for condition in self._trigger_conditions)
        if self.keep[0] == "fraction":
            requires_profile = True
        if requires_profile and self._get_context_length() is None:
            msg = (
                "Model context window information is required to use fractional token limits. "
                "Please either:\n"
                "1. Set max_context_window parameter (e.g., max_context_window=128000 for 128K models)\n"
                "2. Use absolute token counts instead (e.g., trigger=('tokens', 100000))\n"
                "3. Use a model with profile information: ChatModel(..., profile={'max_input_tokens': ...})"
            )
            raise ValueError(msg)

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """在模型调用前检查是否需要压缩。"""
        messages = self._ensure_message_ids(state["messages"])

        # 使用倒序扫描获取真实 token 使用量
        total_tokens = self._get_current_token_usage(messages)

        if not self._should_summarize(messages, total_tokens):
            return None

        cutoff_index = self._determine_cutoff_index(messages)
        if cutoff_index <= 0:
            return None

        # 分离消息（自动保留原始 SystemMessage）
        messages_to_summarize, preserved_messages, original_system = self._partition_messages(messages, cutoff_index)

        # 同步生成摘要
        summary = self._create_summary(messages_to_summarize)
        summary_message = self._build_new_messages(summary)

        # 构建新的消息列表
        new_messages: list[AnyMessage] = []
        if original_system is not None:
            new_messages.append(original_system)  # 保留原始系统提示词
        new_messages.extend(summary_message)       # AU2 压缩摘要
        new_messages.extend(preserved_messages)    # 保留的最近消息

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
            ],
        }

    async def abefore_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """异步版本：在模型调用前检查是否需要压缩。"""
        messages = self._ensure_message_ids(state["messages"])

        # 使用倒序扫描获取真实 token 使用量
        total_tokens = self._get_current_token_usage(messages)

        if not self._should_summarize(messages, total_tokens):
            return None

        cutoff_index = self._determine_cutoff_index(messages)
        if cutoff_index <= 0:
            return None

        # 分离消息（自动保留原始 SystemMessage）
        messages_to_summarize, preserved_messages, original_system = self._partition_messages(messages, cutoff_index)

        # 异步生成摘要
        summary = await self._acreate_summary(messages_to_summarize)
        summary_message = self._build_new_messages(summary)

        # 构建新的消息列表
        new_messages: list[AnyMessage] = []
        if original_system is not None:
            new_messages.append(original_system)  # 保留原始系统提示词
        new_messages.extend(summary_message)       # AU2 压缩摘要
        new_messages.extend(preserved_messages)    # 保留的最近消息

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
            ],
        }

    def _get_current_token_usage(self, messages: list[AnyMessage]) -> int:
        """倒序扫描获取最新的 token 使用情况。
        
        从最新消息开始往前查找，找到第一个包含 usage 信息的 AIMessage 即停止。
        时间复杂度：O(k)，k 通常为 1-3
        
        如果找不到 usage 信息，回退到启发式估算。
        """
        # 倒序扫描，从最新消息开始找
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, AIMessage):
                usage = self._extract_usage_from_message(msg)
                if usage.get("total", 0) > 0:
                    return usage["total"]

        # 找不到 usage 信息，回退到启发式估算
        return self.token_counter(messages)

    def _extract_usage_from_message(self, message: AIMessage) -> dict[str, int]:
        """从 AIMessage 中提取 token 使用信息。"""
        usage = {"input": 0, "output": 0, "total": 0}

        # 尝试从 usage_metadata 提取（LangChain 新版本）
        if hasattr(message, "usage_metadata") and message.usage_metadata:
            usage["input"] = message.usage_metadata.get("input_tokens", 0)
            usage["output"] = message.usage_metadata.get("output_tokens", 0)
            usage["total"] = message.usage_metadata.get("total_tokens", 0)
            return usage

        # 尝试从 response_metadata 提取（兼容旧版本）
        if hasattr(message, "response_metadata") and message.response_metadata:
            token_usage = message.response_metadata.get("token_usage", {})
            usage["input"] = token_usage.get("prompt_tokens", 0)
            usage["output"] = token_usage.get("completion_tokens", 0)
            usage["total"] = token_usage.get("total_tokens", 0)

        return usage

    def _should_summarize(self, messages: list[AnyMessage], total_tokens: int) -> bool:
        """判断是否应该触发压缩。
        
        考虑预留输出 token 空间（max_output_tokens）。
        """
        if not self._trigger_conditions:
            return False

        max_output_tokens = self._get_max_output()
        for kind, value in self._trigger_conditions:
            if kind == "messages" and len(messages) >= value:
                return True
            if kind == "tokens":
                # 考虑预留输出空间
                available = value - max_output_tokens
                if available > 0 and total_tokens >= available:
                    return True
            if kind == "fraction":
                max_input_tokens = self._get_context_length()
                if max_input_tokens is None:
                    continue
                # 考虑预留输出空间
                available = max_input_tokens - max_output_tokens
                threshold = int(available * value)
                if threshold <= 0:
                    threshold = 1
                if total_tokens >= threshold:
                    return True
        return False

    def _determine_cutoff_index(self, messages: list[AnyMessage]) -> int:
        """确定切分点，同时确保不会切断 AI ↔ Tool 消息对。"""
        kind, value = self.keep
        if kind in {"tokens", "fraction"}:
            token_based_cutoff = self._find_token_based_cutoff(messages)
            if token_based_cutoff is not None:
                return token_based_cutoff
            return self._find_safe_cutoff(messages, _DEFAULT_MESSAGES_TO_KEEP)
        return self._find_safe_cutoff(messages, cast("int", value))

    def _find_token_based_cutoff(self, messages: list[AnyMessage]) -> int | None:
        """基于 token 数量找到切分点。"""
        if not messages:
            return 0

        kind, value = self.keep
        if kind == "fraction":
            max_input_tokens = self._get_context_length()
            max_output_tokens = self._get_max_output()
            if max_input_tokens is None:
                return None
            target_token_count = int((max_input_tokens - max_output_tokens) * value)
        elif kind == "tokens":
            target_token_count = int(value)
        else:
            return None

        if target_token_count <= 0:
            target_token_count = 1

        if self.token_counter(messages) <= target_token_count:
            return 0

        # 二分查找
        left, right = 0, len(messages)
        cutoff_candidate = len(messages)
        max_iterations = len(messages).bit_length() + 1

        for _ in range(max_iterations):
            if left >= right:
                break
            mid = (left + right) // 2
            if self.token_counter(messages[mid:]) <= target_token_count:
                cutoff_candidate = mid
                right = mid
            else:
                left = mid + 1

        if cutoff_candidate == len(messages):
            cutoff_candidate = left
        if cutoff_candidate >= len(messages):
            cutoff_candidate = len(messages) - 1 if len(messages) > 1 else 0

        # 确保切分点安全
        for i in range(cutoff_candidate, -1, -1):
            if self._is_safe_cutoff_point(messages, i):
                return i

        return 0

    def _find_safe_cutoff(self, messages: list[AnyMessage], messages_to_keep: int) -> int:
        """找到安全的切分点，确保不会分离 AI ↔ Tool 消息对。"""
        if len(messages) <= messages_to_keep:
            return 0

        target_cutoff = len(messages) - messages_to_keep

        for i in range(target_cutoff, -1, -1):
            if self._is_safe_cutoff_point(messages, i):
                return i

        return 0

    def _is_safe_cutoff_point(self, messages: list[AnyMessage], cutoff_index: int) -> bool:
        """检查切分点是否安全（不会分离 AI ↔ Tool 消息对）。"""
        if cutoff_index >= len(messages):
            return True

        search_start = max(0, cutoff_index - _SEARCH_RANGE_FOR_TOOL_PAIRS)
        search_end = min(len(messages), cutoff_index + _SEARCH_RANGE_FOR_TOOL_PAIRS)

        for i in range(search_start, search_end):
            if not self._has_tool_calls(messages[i]):
                continue

            tool_call_ids = self._extract_tool_call_ids(cast("AIMessage", messages[i]))
            if self._cutoff_separates_tool_pair(messages, i, cutoff_index, tool_call_ids):
                return False

        return True

    def _has_tool_calls(self, message: AnyMessage) -> bool:
        """检查消息是否是带有 tool_calls 的 AI 消息。"""
        return (
            isinstance(message, AIMessage)
            and hasattr(message, "tool_calls")
            and bool(message.tool_calls)
        )

    def _extract_tool_call_ids(self, ai_message: AIMessage) -> set[str]:
        """提取 AI 消息中的 tool_call_id。"""
        tool_call_ids = set()
        for tc in ai_message.tool_calls:
            call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if call_id is not None:
                tool_call_ids.add(call_id)
        return tool_call_ids

    def _cutoff_separates_tool_pair(
        self,
        messages: list[AnyMessage],
        ai_message_index: int,
        cutoff_index: int,
        tool_call_ids: set[str],
    ) -> bool:
        """检查切分点是否会分离 AI 消息和对应的 Tool 消息。"""
        for j in range(ai_message_index + 1, len(messages)):
            message = messages[j]
            if isinstance(message, ToolMessage) and message.tool_call_id in tool_call_ids:
                ai_before_cutoff = ai_message_index < cutoff_index
                tool_before_cutoff = j < cutoff_index
                if ai_before_cutoff != tool_before_cutoff:
                    return True
        return False

    def _partition_messages(
        self,
        messages: list[AnyMessage],
        cutoff_index: int,
    ) -> tuple[list[AnyMessage], list[AnyMessage], SystemMessage | None]:
        """分割消息为需要压缩的部分和需要保留的部分。
        
        自动识别并保留原始 SystemMessage（如果存在）。
        
        Returns:
            (messages_to_summarize, preserved_messages, original_system_message)
        """
        # 检查第一条消息是否是 SystemMessage
        original_system_message = None
        start_index = 0

        if messages and isinstance(messages[0], SystemMessage):
            original_system_message = messages[0]
            start_index = 1
            # 确保 cutoff_index 不会把 SystemMessage 也放入压缩
            cutoff_index = max(start_index, cutoff_index)

        messages_to_summarize = messages[start_index:cutoff_index]
        preserved_messages = messages[cutoff_index:]

        return messages_to_summarize, preserved_messages, original_system_message

    def _format_messages_for_compression(self, messages: list[AnyMessage]) -> str:
        """将消息格式化为适合压缩的文本。"""
        formatted = []
        for i, msg in enumerate(messages):
            role = "Unknown"
            if isinstance(msg, HumanMessage):
                role = "User"
            elif isinstance(msg, AIMessage):
                role = "Assistant"
            elif isinstance(msg, SystemMessage):
                role = "System"
            elif isinstance(msg, ToolMessage):
                role = f"Tool({getattr(msg, 'name', 'unknown')})"

            content = msg.content if hasattr(msg, "content") else str(msg)
            if isinstance(content, list):
                # 处理多模态内容
                content = str(content)
            formatted.append(f"[{i+1}] {role}: {content}")

        return "\n\n".join(formatted)

    def _trim_messages_for_summary(self, messages: list[AnyMessage]) -> str:
        """裁剪消息以适应压缩模型的输入限制。
        
        使用 token_counter 精确计算，优先保留最近的消息。
        """
        if self.trim_tokens_to_summarize is None:
            return self._format_messages_for_compression(messages)

        # 检查是否需要裁剪
        total_tokens = self.token_counter(messages)
        if total_tokens <= self.trim_tokens_to_summarize:
            return self._format_messages_for_compression(messages)

        # 从后往前选择消息，直到达到 token 限制（保留最近的消息）
        selected_messages: list[AnyMessage] = []
        current_tokens = 0

        for msg in reversed(messages):
            msg_tokens = self.token_counter([msg])
            if current_tokens + msg_tokens > self.trim_tokens_to_summarize:
                break
            selected_messages.insert(0, msg)
            current_tokens += msg_tokens

        # 如果一条消息都选不了，至少选最后一条
        if not selected_messages and messages:
            selected_messages = [messages[-1]]

        if len(selected_messages) < len(messages):
            truncated_count = len(messages) - len(selected_messages)
            formatted = f"[{truncated_count} earlier messages truncated]\n\n" + self._format_messages_for_compression(selected_messages)
        else:
            formatted = self._format_messages_for_compression(selected_messages)

        return formatted

    def _build_new_messages(self, summary: str) -> list[SystemMessage]:
        """构建新的消息列表（包含压缩摘要）。"""
        return [
            SystemMessage(content=f"{self.AU2_SUMMARY_PREFIX}\n\n{summary}"),
        ]

    def _ensure_message_ids(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """确保所有消息都有唯一 ID。
        
        返回处理后的消息列表（可能包含复制的消息对象）。
        """
        result = []
        for msg in messages:
            if msg.id is None:
                # 尝试创建副本避免修改原对象
                try:
                    # Pydantic V2 使用 model_copy
                    if hasattr(msg, "model_copy"):
                        msg = msg.model_copy(update={"id": str(uuid.uuid4())})
                    else:
                        # 旧版本 Pydantic 使用 copy
                        msg = msg.copy(update={"id": str(uuid.uuid4())})
                except (AttributeError, TypeError):
                    # 如果不支持 copy，直接修改（BaseMessage 通常支持）
                    msg.id = str(uuid.uuid4())
            result.append(msg)
        return result

    def _create_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        """同步生成摘要。"""
        if not messages_to_summarize:
            return "No previous conversation history."

        original_tokens = self.token_counter(messages_to_summarize)
        formatted = self._trim_messages_for_summary(messages_to_summarize)

        try:
            response = self.model.invoke(self.compression_prompt.format(messages=formatted))
            summary = response.content.strip() if hasattr(response, "content") else str(response).strip()

            # 计算压缩统计
            compressed_tokens = self.token_counter([SystemMessage(content=summary)])
            ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0

            logging.info(
                f"AU2 compression: {original_tokens} → {compressed_tokens} tokens "
                f"({ratio:.1%} of original, saved {original_tokens - compressed_tokens} tokens)",
            )

            return summary
        except Exception as e:
            logging.exception(f"AU2 compression failed: {e}")
            # 返回错误提示，让 Agent 知道压缩失败了
            return f"[Compression failed: {e!s}. Retaining recent context only.]"

    async def _acreate_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        """异步生成摘要。"""
        if not messages_to_summarize:
            return "No previous conversation history."

        original_tokens = self.token_counter(messages_to_summarize)
        formatted = self._trim_messages_for_summary(messages_to_summarize)

        try:
            response = await self.model.ainvoke(self.compression_prompt.format(messages=formatted))
            summary = response.content.strip() if hasattr(response, "content") else str(response).strip()

            # 计算压缩统计
            compressed_tokens = self.token_counter([SystemMessage(content=summary)])
            ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0

            logging.info(
                f"AU2 compression: {original_tokens} → {compressed_tokens} tokens "
                f"({ratio:.1%} of original, saved {original_tokens - compressed_tokens} tokens)",
            )

            return summary
        except Exception as e:
            logging.exception(f"AU2 compression failed: {e}")
            return f"[Compression failed: {e!s}. Retaining recent context only.]"

    def _get_context_length(self) -> int | None:
        """获取模型的最大输入 token 限制。

        优先级：
        1. max_context_window 参数（用户显式设置）
        2. 模型的 profile.max_input_tokens
        """
        # 优先使用用户设置的 max_context_window
        if self.max_context_window is not None:
            return self.max_context_window

        # 尝试从模型 profile 获取
        try:
            profile = self.model.profile
        except AttributeError:
            return None

        if not isinstance(profile, Mapping):
            return None

        max_input_tokens = profile.get("max_input_tokens")
        self.max_context_window = max_input_tokens

        return max_input_tokens

    def _get_max_output(self) -> int | None:
        """获取模型的最大输入 token 限制。

        优先级：
        1. max_context_window 参数（用户显式设置）
        2. 模型的 profile.max_input_tokens
        """
        # 优先使用用户设置的 max_context_window
        if self.max_output_tokens is not None:
            return self.max_output_tokens

        # 尝试从模型 profile 获取
        try:
            profile = self.model.profile
        except AttributeError:
            return None

        if not isinstance(profile, Mapping):
            return None

        max_output_tokens = profile.get("max_output_tokens")
        self.max_output_tokens = max_output_tokens

        return max_output_tokens
    
    def _validate_context_size(self, context: ContextSize, parameter_name: str) -> ContextSize:
        """验证上下文大小配置。"""
        kind, value = context
        if kind == "fraction":
            if not 0 < value <= 1:
                msg = f"Fractional {parameter_name} values must be between 0 and 1, got {value}."
                raise ValueError(msg)
        elif kind in {"tokens", "messages"}:
            if value <= 0:
                msg = f"{parameter_name} thresholds must be greater than 0, got {value}."
                raise ValueError(msg)
        else:
            msg = f"Unsupported context size type {kind} for {parameter_name}."
            raise ValueError(msg)
        return context
