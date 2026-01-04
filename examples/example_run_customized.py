"""
完整示例：使用自定义 DeepAgents
- 支持多轮连续对话
- 使用 AU2 8段式结构化压缩（更适合代码开发场景）
- 集成 Tavily 搜索
- FilesystemBackend 真实文件读写
- SubAgent 子代理
- 从 .env 读取配置
"""

import os
import uuid
from typing import Literal
from dotenv import load_dotenv

from tavily import TavilyClient
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from deepagents.backends import FilesystemBackend

from agent_templates import create_deep_agent_customized

# 加载 .env 文件中的环境变量
load_dotenv()

# ============ 从 .env 读取配置 ============
# API 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")

# LLM 参数
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192"))

# Agent 配置
DEFAULT_CONTEXT_LIMIT = int(os.getenv("DEFAULT_CONTEXT_LIMIT", "128000"))

# Tavily 配置
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# 工作目录（用于文件操作）
# 注意：直接使用 "temp"，不要带 "./"
WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# 初始化 Tavily 客户端
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None


# 网络搜索工具
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """运行网络搜索。

    Args:
        query: 搜索查询
        max_results: 最大结果数量
        topic: 搜索主题类型 (general/news/finance)
        include_raw_content: 是否包含原始内容
    """
    if tavily_client is None:
        return {"error": "Tavily API key not configured"}
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


# ============ 自定义工具 ============


@tool
def get_current_time() -> str:
    """获取当前时间。涉及到时间有关的操作，都需要先查询时间"""
    from datetime import datetime

    now = datetime.now()
    return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"


# ============ 定义子代理 ============
# 研究子代理：专门用于网络搜索
research_subagent = {
    "name": "researcher",
    "description": "专门用于网络搜索和研究问题的子代理。当需要搜索互联网获取最新信息时使用。",
    "system_prompt": """你是一个专业的研究助手。你的任务是：
1. 使用 internet_search 工具搜索相关信息
2. 整理并总结搜索结果
3. 提供清晰、准确的研究报告

请始终保持客观和准确。""",
    "tools": [internet_search],
}


# 使用 OpenAI 兼容接口调用模型（从 .env 读取配置）
model = init_chat_model(
    model=f"openai:{OPENAI_MODEL}",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,
)

# 系统提示
system_prompt = """你是一个功能强大的 AI 助手，具有以下能力：

## 核心能力
1. **任务规划**: 使用 write_todos 工具来规划和跟踪复杂任务
2. **文件管理**: 使用 ls, read_file, write_file, edit_file 工具管理真实文件系统
3. **子代理调用**: 使用 task 工具调用专门的子代理：
   - `researcher`: 用于网络搜索和研究
4. **时间查询**: 使用 get_current_time 获取当前时间

## 文件系统
- 当前目录就是工作目录（根目录 /）
- 直接使用文件名保存，如 write_file("/report.md", content)
- 不要再创建 workspace 子目录
- 使用 ls / 查看当前目录内容

## 工作流程
1. 收到复杂任务时，先使用 write_todos 制定计划
2. 按计划逐步执行，更新任务状态
3. 需要网络搜索时，调用 researcher 子代理
4. 需要保存结果时，直接写入根目录
"""

# 创建 FilesystemBackend（真实文件系统）
filesystem_backend = FilesystemBackend(
    root_dir=WORKSPACE_DIR,
    virtual_mode=True,  # 安全模式，限制在 workspace 目录内
)

# 创建 checkpointer 用于保存对话状态
checkpointer = MemorySaver()

# 创建自定义 deep agent
agent = create_deep_agent_customized(
    model=model,
    tools=[get_current_time],  # 主代理的工具
    system_prompt=system_prompt,
    checkpointer=checkpointer,  # 启用对话记忆
    backend=filesystem_backend,  # 真实文件系统
    subagents=[research_subagent],  # 子代理
    # AU2 压缩参数（从 .env 读取）
    max_context_window=DEFAULT_CONTEXT_LIMIT,  # 模型输入上限
    max_output_tokens=LLM_MAX_TOKENS,  # 模型输出上限
    compression_trigger=0.80,  # 80% 时触发压缩
    messages_to_keep=5,  # 保留最近 5 条消息
)


# 是否显示子智能体内部细节
SHOW_SUBAGENT_DETAILS = True

# 是否使用同步模式（invoke 而不是 astream）
USE_SYNC_MODE = False

# 流式模式: "tokens" = token级别流式, "nodes" = 节点级别流式
STREAM_MODE = "tokens"


# 同步模式响应
def sync_response(user_input: str, config: dict):
    """同步调用 agent，等待完整响应"""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
    )

    # 打印所有消息
    for msg in result.get("messages", []):
        msg_type = getattr(msg, "type", type(msg).__name__)
        content = getattr(msg, "content", "")

        if msg_type == "human":
            continue  # 跳过用户消息

        if msg_type == "ai":
            if content:
                print(f"\n💬 AI: {content}")

            tool_calls = getattr(msg, "tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    print(
                        f"\n🔧 调用工具: {tc.get('name', 'unknown')} | 参数: {tc.get('args', {})}"
                    )

        elif msg_type == "tool":
            tool_name = getattr(msg, "name", "unknown")
            display_content = (
                str(content)[:300] + "..." if len(str(content)) > 300 else content
            )
            print(f"\n📦 [{tool_name}]: {display_content}")


# Token 级别流式输出
async def stream_tokens_response(user_input: str, config: dict):
    """Token 级别流式输出，每个 token 实时显示"""
    print("\n💬 AI: ", end="", flush=True)

    # 用于累积工具调用信息（因为参数是分块传来的）
    pending_tool_calls: dict[str, dict] = {}
    printed_tool_calls: set[str] = set()  # 已打印的工具调用

    try:
        async for msg_chunk, metadata in agent.astream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode="messages",
        ):
            msg_type = getattr(msg_chunk, "type", type(msg_chunk).__name__)

            if msg_type == "AIMessageChunk":
                content = getattr(msg_chunk, "content", "")
                if content:
                    print(content, end="", flush=True)

                # 累积工具调用信息（参数是分块传来的）
                tool_call_chunks = getattr(msg_chunk, "tool_call_chunks", [])
                for tc in tool_call_chunks:
                    tc_id = tc.get("id") or str(tc.get("index", 0))
                    if tc_id not in pending_tool_calls:
                        pending_tool_calls[tc_id] = {"name": "", "args": ""}
                    if tc.get("name"):
                        pending_tool_calls[tc_id]["name"] = tc["name"]
                    if tc.get("args"):
                        pending_tool_calls[tc_id]["args"] += tc["args"]

                # 检查完整的工具调用
                tool_calls = getattr(msg_chunk, "tool_calls", [])
                for tc in tool_calls:
                    tc_id = tc.get("id", "")
                    name = tc.get("name", "")
                    args = tc.get("args", {})

                    # 只打印有名字且未打印过的工具调用
                    if name and tc_id not in printed_tool_calls:
                        printed_tool_calls.add(tc_id)
                        # 格式化参数显示
                        if isinstance(args, dict) and args:
                            args_items = []
                            for k, v in args.items():
                                v_str = (
                                    repr(v)
                                    if len(repr(v)) < 50
                                    else repr(v)[:47] + "..."
                                )
                                args_items.append(f"{k}={v_str}")
                            args_str = ", ".join(args_items)
                            print(
                                f"\n\n🔧 调用工具: {name}\n   参数: {args_str}",
                                flush=True,
                            )
                        else:
                            print(f"\n\n🔧 调用工具: {name}", flush=True)

            elif msg_type == "tool":
                tool_name = getattr(msg_chunk, "name", "unknown")
                content = getattr(msg_chunk, "content", "")
                display_content = (
                    str(content)[:300] + "..." if len(str(content)) > 300 else content
                )
                print(f"\n\n📦 [{tool_name}]: {display_content}", flush=True)
                print("\n💬 AI: ", end="", flush=True)

        print()
    except Exception as e:
        error_str = str(e).lower()
        if "connection" in error_str or "timeout" in error_str:
            print(f"\n\n⚠️ 网络连接问题: {e}")
        else:
            print(f"\n\n❌ Token 流式错误: {e}")


# 节点级别流式输出
async def stream_response(user_input: str, config: dict):
    """流式输出 agent 响应，支持显示子智能体内部细节"""

    try:
        if SHOW_SUBAGENT_DETAILS:
            async for chunk in agent.astream(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                stream_mode="updates",
                subgraphs=True,
            ):
                namespace, update = chunk
                is_subagent = len(namespace) > 0
                prefix = "    🔹 [子代理] " if is_subagent else ""

                if "model" in update:
                    messages = update["model"].get("messages", [])
                    for msg in messages:
                        msg_type = getattr(msg, "type", type(msg).__name__)
                        content = getattr(msg, "content", "")

                        if msg_type == "ai":
                            if content:
                                print(f"\n{prefix}💬 AI: {content}")

                            tool_calls = getattr(msg, "tool_calls", [])
                            if tool_calls:
                                for tc in tool_calls:
                                    print(
                                        f"\n{prefix}🔧 调用工具: {tc.get('name', 'unknown')} | 参数: {tc.get('args', {})}"
                                    )

                if "tools" in update:
                    messages = update["tools"].get("messages", [])
                    for msg in messages:
                        tool_name = getattr(msg, "name", "unknown")
                        content = getattr(msg, "content", "")
                        display_content = (
                            str(content)[:300] + "..."
                            if len(str(content)) > 300
                            else content
                        )
                        print(f"\n{prefix}📦 [{tool_name}]: {display_content}")
        else:
            async for chunk in agent.astream(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                stream_mode="values",
            ):
                if "messages" in chunk:
                    msg = chunk["messages"][-1]
                    msg_type = getattr(msg, "type", type(msg).__name__)
                    content = getattr(msg, "content", "")

                    if msg_type == "ai":
                        if content:
                            print(f"\n💬 AI: {content}")

                        tool_calls = getattr(msg, "tool_calls", [])
                        if tool_calls:
                            for tc in tool_calls:
                                print(
                                    f"\n🔧 调用工具: {tc.get('name', 'unknown')} | 参数: {tc.get('args', {})}"
                                )

                    elif msg_type == "tool":
                        tool_name = getattr(msg, "name", "unknown")
                        display_content = (
                            str(content)[:300] + "..."
                            if len(str(content)) > 300
                            else content
                        )
                        print(f"\n📦 [{tool_name}]: {display_content}")
    except Exception as e:
        error_msg = str(e)
        if "tool_calls" in error_msg and "tool messages" in error_msg:
            print("\n⚠️ 对话历史损坏，请输入 'new' 开始新对话")
        else:
            raise


# 多轮对话主循环
async def chat_loop():
    """交互式多轮对话"""
    global SHOW_SUBAGENT_DETAILS, USE_SYNC_MODE, STREAM_MODE

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 60)
    print("🤖 DeepAgents 多轮对话模式 (自定义版)")
    print("=" * 60)
    print(f"📌 对话线程 ID: {thread_id}")
    print(f"📁 工作目录: {WORKSPACE_DIR}")
    print(f"🔧 模型: {OPENAI_MODEL}")
    print(f"📊 上下文窗口: {DEFAULT_CONTEXT_LIMIT:,} tokens")
    print(f"📊 最大输出: {LLM_MAX_TOKENS:,} tokens")
    print(f"🌡️ Temperature: {LLM_TEMPERATURE}")
    print("-" * 60)
    print("💡 输入 'quit' 或 'exit' 退出对话")
    print("💡 输入 'new' 开始新对话")
    print("💡 输入 'toggle' 切换子代理细节显示")
    print("💡 输入 'sync' 切换同步/流式模式")
    print("💡 输入 'stream' 切换 token/节点 级别流式")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n👤 你: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 再见！")
                break

            if user_input.lower() == "new":
                thread_id = str(uuid.uuid4())
                config = {"configurable": {"thread_id": thread_id}}
                print(f"\n🔄 已开始新对话，线程 ID: {thread_id}")
                continue

            if user_input.lower() == "toggle":
                SHOW_SUBAGENT_DETAILS = not SHOW_SUBAGENT_DETAILS
                status = "开启" if SHOW_SUBAGENT_DETAILS else "关闭"
                print(f"\n🔄 子代理细节显示已{status}")
                continue

            if user_input.lower() == "sync":
                USE_SYNC_MODE = not USE_SYNC_MODE
                mode = "同步" if USE_SYNC_MODE else "流式"
                print(f"\n🔄 已切换到{mode}模式")
                continue

            if user_input.lower() == "stream":
                STREAM_MODE = "nodes" if STREAM_MODE == "tokens" else "tokens"
                mode_name = "Token级别" if STREAM_MODE == "tokens" else "节点级别"
                print(f"\n🔄 已切换到{mode_name}流式模式")
                continue

            if USE_SYNC_MODE:
                sync_response(user_input, config)
            elif STREAM_MODE == "tokens":
                await stream_tokens_response(user_input, config)
            else:
                await stream_response(user_input, config)

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(chat_loop())
