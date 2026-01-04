"""
使用 create_deep_agent_customized 实现 Open Deep Research Agent

这个实现基于 LangGraph 的 open_deep_research 项目，使用 deepagents 框架重新实现。

核心架构：
1. 主代理 (Supervisor): 分析研究问题，制定研究计划，委派子代理执行研究
2. 研究子代理 (Researcher): 执行具体的网络搜索和信息收集
3. 最终报告生成：汇总所有研究结果，生成结构化报告

特点：
- 支持多轮对话和澄清问题
- 支持并行研究多个子主题
- 自动压缩和管理上下文
- 生成带引用的专业研究报告
"""

import os
import uuid
import asyncio
from datetime import datetime
from typing import Literal
import requests

from dotenv import load_dotenv
from tavily import TavilyClient, AsyncTavilyClient
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from deepagents.backends import FilesystemBackend

from agent_templates import create_deep_agent_customized

load_dotenv()

# ============ 配置 ============
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

# 博查配置
USE_BOCHAI_SEARCH = os.getenv("USE_BOCHAI_SEARCH", "false").lower() == "true"
BOCHAI_API_KEY = os.getenv("BOCHAI_API_KEY")
BOCHAI_BASE_URL = os.getenv("BOCHAI_BASE_URL", "https://api.bocha.cn/v1/web-search")

# 研究配置
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
MAX_CONCURRENT_RESEARCH = int(os.getenv("MAX_CONCURRENT_RESEARCH", "3"))

# 工作目录
WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(WORKSPACE_DIR, exist_ok=True)


# ============ 辅助函数 ============
def get_today_str() -> str:
    """获取今天的日期字符串"""
    return datetime.now().strftime("%Y-%m-%d")


# ============ Prompts ============
CLARIFY_WITH_USER_PROMPT = """分析用户的研究请求，判断是否需要澄清问题。

用户消息:
{messages}

今天日期: {date}

评估标准:
1. 研究主题是否清晰明确？
2. 是否存在模糊的缩写或专业术语？
3. 研究范围是否合理？

如果需要澄清:
- 提出简洁明确的问题
- 使用项目符号列出需要澄清的点

如果不需要澄清:
- 确认理解用户的研究需求
- 简要总结研究要点
- 表示即将开始研究
"""

RESEARCH_BRIEF_PROMPT = """将用户的研究请求转化为详细的研究简报。

用户消息:
{messages}

今天日期: {date}

要求:
1. 最大化具体性和细节
2. 包含用户提到的所有偏好和要求
3. 对未指定的必要维度保持开放
4. 使用第一人称
5. 如果有特定来源偏好，请说明
"""

SUPERVISOR_SYSTEM_PROMPT = """你是一个研究主管。你的工作是通过调用子代理来进行深度研究。今天日期: {date}

## 任务
你需要分析用户的研究问题，将其分解为可管理的子任务，然后委派给专门的研究子代理执行。

## 可用工具
1. **task** - 调用研究子代理执行具体搜索任务。格式: task("researcher", "具体研究主题")
2. **write_todos** - 制定研究计划
3. **think** - 反思和规划（每次调用子代理前后都应使用）
4. **write_file** - 保存研究报告到文件。格式: write_file("/报告名.md", 报告内容)
5. **ls** - 列出目录内容
6. **read_file** - 读取文件内容

## 文件系统
- 当前工作目录是根目录 `/`
- 研究报告必须保存到文件，使用 `write_file("/研究报告_主题.md", 内容)` 保存
- 文件名应该简洁明了，包含研究主题

## 工作流程

### 第一步：分析与规划
收到研究问题后，首先使用 think 工具分析：
- 这个问题需要哪些方面的信息？
- 是否可以分解为独立的子研究？
- 需要多少个并行研究？

### 第二步：委派研究
- 对于简单查询：使用 1 个子代理
- 对于比较类查询：为每个比较对象分配子代理
- 对于复杂查询：将问题分解为 2-{max_concurrent} 个独立子任务

### 第三步：评估结果
每次子代理返回后，使用 think 工具评估：
- 获得了哪些关键信息？
- 还缺少什么？
- 是否足够回答问题？

### 第四步：生成并保存报告
当收集到足够信息后：
1. 撰写最终研究报告：
   - 使用清晰的结构（标题、章节、小节）
   - 包含所有相关发现和引用
   - 使用 [标题](URL) 格式引用来源
   - 在末尾列出所有来源
2. **必须使用 write_file 工具将报告保存到文件**：
   - 格式：`write_file("/研究报告_主题名称.md", 报告内容)`
   - 例如：`write_file("/研究报告_AI模型对比.md", "# AI模型对比...")`
3. 向用户确认报告已保存，并给出简要总结

## 限制
- 最多并行 {max_concurrent} 个研究任务
- 最多进行 {max_iterations} 轮研究迭代
- 当有足够信息回答问题时立即停止
- **完成研究后必须保存报告到文件**

## 输出格式
最终报告应该包含：
1. 概述/引言
2. 主要发现（按主题组织）
3. 分析和结论
4. 来源列表
"""

RESEARCHER_SYSTEM_PROMPT = """你是一个专业的研究助手，负责执行具体的信息搜索任务。今天日期: {date}

## 任务
使用提供的搜索工具收集关于指定主题的信息。

## 可用工具
1. **{search_tool_name}** - 网络搜索，获取最新信息
2. **think** - 反思搜索结果，规划下一步
3. **write_file** - 保存研究发现到文件。格式: write_file("/research_主题.md", 内容)

## 文件系统
- 工作目录是根目录 `/`
- 完成研究后，将研究发现保存到文件
- 文件名格式：`/research_主题关键词.md`

## 工作流程

### 搜索策略
1. 先进行广泛搜索，了解主题概况
2. 根据初步结果进行针对性搜索
3. 每次搜索后用 think 评估收获

### 搜索预算
- 简单查询：2-3 次搜索
- 复杂查询：最多 5 次搜索
- 找到 3+ 个相关来源后考虑停止

### 停止条件
- 能够全面回答研究问题
- 已有 3 个以上相关来源
- 最近 2 次搜索返回重复信息

## 输出要求
完成搜索后：
1. 整理发现：
   - 列出所有查询和工具调用
   - 详细记录所有发现（保留原始信息）
   - 为每个来源提供完整引用
   - 不要丢失任何相关信息
2. **必须使用 write_file 保存研究发现**：
   - 格式：`write_file("/research_主题关键词.md", 研究发现内容)`
   - 例如：`write_file("/research_deepseek.md", "# Deepseek研究发现\n...")`
3. 返回简要总结给主代理
"""

FINAL_REPORT_PROMPT = """基于所有收集的研究发现，创建一份全面的研究报告。

研究简报:
{research_brief}

研究发现:
{findings}

今天日期: {date}

## 报告要求

### 结构
1. 使用清晰的标题层级（# 标题，## 章节，### 小节）
2. 根据内容类型选择合适的结构：
   - 比较类：概述 → 各项介绍 → 对比分析 → 结论
   - 列表类：直接列表或分项介绍
   - 综述类：概述 → 各概念详解 → 总结

### 内容
1. 使用简洁清晰的语言
2. 包含具体事实和见解
3. 各章节应足够详细
4. 适当使用项目符号

### 引用规则
1. 为每个 URL 分配唯一编号
2. 在文中使用 [1]、[2] 等标注
3. 文末列出所有来源：
   [1] 来源标题: URL
   [2] 来源标题: URL

### 语言
- 使用与用户输入相同的语言撰写
- 如果用户使用中文，报告使用中文
- 如果用户使用英文，报告使用英文

## 注意事项
- 不要自我引用（不说"我"、"本报告"等）
- 不要评论自己在做什么
- 直接呈现研究内容
"""


# ============ 工具定义 ============

# 初始化 Tavily 客户端
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
async_tavily_client = (
    AsyncTavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
)


@tool
def internet_search(
    query: str,
    max_results: int = MAX_SEARCH_RESULTS,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> dict:
    """使用 Tavily 执行网络搜索。"""
    if tavily_client is None:
        return {
            "error": "Tavily API key not configured. Please set TAVILY_API_KEY in .env"
        }

    try:
        result = tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )

        formatted_results = []
        for i, item in enumerate(result.get("results", []), 1):
            formatted_results.append(
                f"--- 来源 {i}: {item.get('title', 'Unknown')} ---\n"
                f"URL: {item.get('url', 'N/A')}\n"
                f"摘要: {item.get('content', 'N/A')}\n"
            )

        return {
            "query": query,
            "results": "\n\n".join(formatted_results)
            if formatted_results
            else "未找到结果",
            "result_count": len(result.get("results", [])),
        }
    except Exception as e:
        return {"error": f"Tavily search failed: {str(e)}"}


@tool
def internet_search_bochai(
    query: str,
    count: int = MAX_SEARCH_RESULTS,
    summary: bool = True,
) -> dict:
    """使用博查接口执行网络搜索。"""
    if not BOCHAI_API_KEY:
        return {
            "error": "BoChai API key not configured. Please set BOCHAI_API_KEY in .env"
        }

    headers = {
        "Authorization": f"{BOCHAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"query": query, "summary": summary, "count": count}

    try:
        response = requests.post(
            BOCHAI_BASE_URL, headers=headers, json=payload, timeout=30
        )
        response.raise_for_status()
        data = response.json()
        items = data.get("data", {}).get("webPages", {}).get("value", [])

        formatted_results = []
        for i, item in enumerate(items, 1):
            formatted_results.append(
                f"--- 来源 {i}: {item.get('name', 'Unknown')} ---\n"
                f"URL: {item.get('url', 'N/A')}\n"
                f"摘要: {item.get('snippet', 'N/A')}\n"
            )

        return {
            "query": query,
            "results": "\n\n".join(formatted_results)
            if formatted_results
            else "未找到结果",
            "result_count": len(items),
        }
    except Exception as e:
        return {"error": f"BoChai search failed: {str(e)}"}


@tool
def think(reflection: str) -> str:
    """反思工具 - 用于战略规划和结果分析。

    在以下情况使用此工具：
    1. 分析研究问题，规划研究策略
    2. 评估搜索结果，决定下一步
    3. 判断是否已收集足够信息

    Args:
        reflection: 你的反思和思考内容

    Returns:
        确认反思已记录
    """
    return f"反思已记录: {reflection}"


@tool
def get_current_time() -> str:
    """获取当前时间。涉及到时间有关的操作，都需要先查询时间"""
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"


# ============ 研究子代理定义 ============
search_tool = internet_search_bochai if USE_BOCHAI_SEARCH else internet_search
# StructuredTool 没有 __name__，使用 name 属性；函数本身也支持 __name__ 作为备选
search_tool_name = getattr(search_tool, "name", None) or getattr(
    search_tool, "__name__", "internet_search"
)

researcher_subagent = {
    "name": "researcher",
    "description": """专业的研究助手，用于执行网络搜索和信息收集。
    
当需要搜索互联网获取信息时使用此代理。它会：
1. 执行多次搜索以全面覆盖主题
2. 整理和格式化搜索结果
3. 提供带引用的研究发现""",
    "system_prompt": RESEARCHER_SYSTEM_PROMPT.format(
        date=get_today_str(),
        search_tool_name=search_tool_name,
    ),
    "tools": [search_tool, think],
}


# ============ 创建模型和代理 ============

# 使用 OpenAI 兼容接口调用模型
model = init_chat_model(
    model=f"openai:{OPENAI_MODEL}",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,
)

# 主代理系统提示
main_system_prompt = SUPERVISOR_SYSTEM_PROMPT.format(
    date=get_today_str(),
    max_concurrent=MAX_CONCURRENT_RESEARCH,
    max_iterations=6,
)

# 创建 FilesystemBackend（用于保存研究报告）
filesystem_backend = FilesystemBackend(
    root_dir=WORKSPACE_DIR,
    virtual_mode=True,
)

# 创建 checkpointer
checkpointer = MemorySaver()

# 创建 Open Deep Research Agent
agent = create_deep_agent_customized(
    model=model,
    tools=[think, get_current_time],  # 主代理工具
    system_prompt=main_system_prompt,
    checkpointer=checkpointer,
    backend=filesystem_backend,
    subagents=[researcher_subagent],  # 研究子代理
    # 压缩参数
    max_context_window=DEFAULT_CONTEXT_LIMIT,
    max_output_tokens=LLM_MAX_TOKENS,
    compression_trigger=0.80,
    messages_to_keep=5,
)


# ============ 运行配置 ============
SHOW_SUBAGENT_DETAILS = True
USE_SYNC_MODE = False
STREAM_MODE = "tokens"


# ============ 响应函数 ============


def sync_response(user_input: str, config: dict):
    """同步调用 agent"""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
    )

    for msg in result.get("messages", []):
        msg_type = getattr(msg, "type", type(msg).__name__)
        content = getattr(msg, "content", "")

        if msg_type == "human":
            continue

        if msg_type == "ai":
            if content:
                print(f"\n💬 AI: {content}")

            tool_calls = getattr(msg, "tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    print(
                        f"\n🔧 工具调用: {tc.get('name', 'unknown')} | 参数: {tc.get('args', {})}"
                    )

        elif msg_type == "tool":
            tool_name = getattr(msg, "name", "unknown")
            display_content = (
                str(content)[:500] + "..." if len(str(content)) > 500 else content
            )
            print(f"\n📦 [{tool_name}]: {display_content}")


async def stream_tokens_response(user_input: str, config: dict):
    """Token 级别流式输出"""
    print("\n💬 AI: ", end="", flush=True)

    pending_tool_calls: dict[str, dict] = {}
    printed_tool_calls: set[str] = set()

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

                tool_call_chunks = getattr(msg_chunk, "tool_call_chunks", [])
                for tc in tool_call_chunks:
                    tc_id = tc.get("id") or str(tc.get("index", 0))
                    if tc_id not in pending_tool_calls:
                        pending_tool_calls[tc_id] = {"name": "", "args": ""}
                    if tc.get("name"):
                        pending_tool_calls[tc_id]["name"] = tc["name"]
                    if tc.get("args"):
                        pending_tool_calls[tc_id]["args"] += tc["args"]

                tool_calls = getattr(msg_chunk, "tool_calls", [])
                for tc in tool_calls:
                    tc_id = tc.get("id", "")
                    name = tc.get("name", "")
                    args = tc.get("args", {})

                    if name and tc_id not in printed_tool_calls:
                        printed_tool_calls.add(tc_id)
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
                                f"\n\n🔧 工具调用: {name}\n   参数: {args_str}",
                                flush=True,
                            )
                        else:
                            print(f"\n\n🔧 工具调用: {name}", flush=True)

            elif msg_type == "tool":
                tool_name = getattr(msg_chunk, "name", "unknown")
                content = getattr(msg_chunk, "content", "")
                display_content = (
                    str(content)[:500] + "..." if len(str(content)) > 500 else content
                )
                print(f"\n\n📦 [{tool_name}]: {display_content}", flush=True)
                print("\n💬 AI: ", end="", flush=True)

        print()
    except Exception as e:
        error_str = str(e).lower()
        if "connection" in error_str or "timeout" in error_str:
            print(f"\n\n⚠️ 网络连接问题: {e}")
        else:
            print(f"\n\n❌ 错误: {e}")


async def stream_response(user_input: str, config: dict):
    """节点级别流式输出"""
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
                prefix = "    🔹 [研究子代理] " if is_subagent else ""

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
                                        f"\n{prefix}🔧 工具调用: {tc.get('name', 'unknown')} | 参数: {tc.get('args', {})}"
                                    )

                if "tools" in update:
                    messages = update["tools"].get("messages", [])
                    for msg in messages:
                        tool_name = getattr(msg, "name", "unknown")
                        content = getattr(msg, "content", "")
                        display_content = (
                            str(content)[:500] + "..."
                            if len(str(content)) > 500
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
                                    f"\n🔧 工具调用: {tc.get('name', 'unknown')} | 参数: {tc.get('args', {})}"
                                )

                    elif msg_type == "tool":
                        tool_name = getattr(msg, "name", "unknown")
                        display_content = (
                            str(content)[:500] + "..."
                            if len(str(content)) > 500
                            else content
                        )
                        print(f"\n📦 [{tool_name}]: {display_content}")
    except Exception as e:
        error_msg = str(e)
        if "tool_calls" in error_msg and "tool messages" in error_msg:
            print("\n⚠️ 对话历史损坏，请输入 'new' 开始新对话")
        else:
            raise


# ============ 主循环 ============


async def chat_loop():
    """交互式研究对话"""
    global SHOW_SUBAGENT_DETAILS, USE_SYNC_MODE, STREAM_MODE

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 70)
    print("🔬 Open Deep Research Agent (基于 DeepAgents)")
    print("=" * 70)
    print(f"📌 对话线程 ID: {thread_id}")
    print(f"📁 工作目录: {WORKSPACE_DIR}")
    print(f"🔧 模型: {OPENAI_MODEL}")
    print(f"📊 上下文窗口: {DEFAULT_CONTEXT_LIMIT:,} tokens")
    print(f"🔍 搜索结果数: {MAX_SEARCH_RESULTS}")
    print(f"🌐 搜索工具: {'BoChai' if USE_BOCHAI_SEARCH else 'Tavily'}")
    print("-" * 70)
    print("💡 使用说明:")
    print("   - 输入研究问题，AI 将自动规划并执行深度研究")
    print("   - 支持中英文输入")
    print("   - 研究报告将自动生成带引用的结构化内容")
    print("-" * 70)
    print("💡 命令:")
    print("   'quit' / 'exit' - 退出")
    print("   'new' - 开始新对话")
    print("   'toggle' - 切换子代理细节显示")
    print("   'sync' - 切换同步/流式模式")
    print("   'stream' - 切换 token/节点级别流式")
    print("=" * 70)

    # 示例提示
    print("\n 示例研究问题:")
    print("   1. 比较 GPT-5.1 和 Claude 4.5 opus 在代码生成方面的能力")
    print("   2. 2025年人工智能领域有哪些重大突破？")
    print()

    while True:
        try:
            user_input = input("\n👤 研究问题: ").strip()

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

            print("\n" + "=" * 70)
            print("🔬 开始深度研究...")
            print("=" * 70)

            if USE_SYNC_MODE:
                sync_response(user_input, config)
            elif STREAM_MODE == "tokens":
                await stream_tokens_response(user_input, config)
            else:
                await stream_response(user_input, config)

            print("\n" + "=" * 70)
            print("✅ 研究完成")
            print("=" * 70)

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback

            traceback.print_exc()


# ============ 快速研究函数 ============


async def quick_research(question: str, save_to_file: bool = True) -> str:
    """快速执行一次研究并返回报告

    Args:
        question: 研究问题
        save_to_file: 是否将报告保存到文件

    Returns:
        研究报告内容
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"🔬 开始研究: {question}")
    print("-" * 50)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )

    # 获取最终回复
    final_response = ""
    for msg in result.get("messages", []):
        msg_type = getattr(msg, "type", type(msg).__name__)
        if msg_type == "ai":
            content = getattr(msg, "content", "")
            if content:
                final_response = content

    if save_to_file and final_response:
        # 生成文件名
        safe_name = "".join(
            c if c.isalnum() or c in " _-" else "_" for c in question[:50]
        )
        filename = f"research_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(WORKSPACE_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# 研究报告: {question}\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            f.write(final_response)

        print(f"\n📄 报告已保存到: {filepath}")

    return final_response


if __name__ == "__main__":
    asyncio.run(chat_loop())
