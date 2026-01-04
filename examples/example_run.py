"""
完整示例：使用 DeepSeek 运行 deepagents
- 支持多轮连续对话
- 集成 Tavily 搜索
- FilesystemBackend 真实文件读写
- SubAgent 子代理
- LangSmith 追踪（不使用环境变量）
"""

import os
import uuid
from typing import Literal

from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

# LangSmith 追踪（不使用环境变量）
from langsmith import Client as LangSmithClient, tracing_context

# 加载 .env 文件中的环境变量
load_dotenv()

# ============ LangSmith 配置 ============
# 直接在代码中配置，无需设置环境变量
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = "deepagents-demo"  # 项目名称，可自定义
LANGSMITH_ENABLED = False  # 默认禁用，国内网络访问 LangSmith 不稳定

# 创建 LangSmith 客户端（增加超时设置）
langsmith_client = LangSmithClient(
    api_key=LANGSMITH_API_KEY,
    api_url="https://api.smith.langchain.com",
    timeout_ms=10000,  # 10秒超时
)

# 初始化 Tavily 客户端
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# 工作目录（用于文件操作）
WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(WORKSPACE_DIR, exist_ok=True)


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


# 使用 OpenAI 兼容接口调用 DeepSeek
# .env 中已配置 OPENAI_API_KEY 和 OPENAI_BASE_URL 指向 DeepSeek
model = init_chat_model(
    model="openai:deepseek-chat",  # 使用 deepseek-chat 模型
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    max_tokens=8192,  # 增加输出 token 限制（DeepSeek 最大支持 8K）
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
- 工作目录: workspace/
- 你可以在这个目录下创建、读取、编辑文件
- 使用 ls 查看目录内容

## 工作流程
1. 收到复杂任务时，先使用 write_todos 制定计划
2. 按计划逐步执行，更新任务状态
3. 需要网络搜索时，调用 researcher 子代理
4. 需要保存结果时，写入文件系统
"""

# 创建 FilesystemBackend（真实文件系统）
filesystem_backend = FilesystemBackend(
    root_dir=WORKSPACE_DIR,
    virtual_mode=True,  # 安全模式，限制在 workspace 目录内
)

# 创建 checkpointer 用于保存对话状态
checkpointer = MemorySaver()

# 创建 deep agent（带 checkpointer、backend、subagents）
agent = create_deep_agent(
    model=model,
    tools=[get_current_time],  # 主代理的工具
    system_prompt=system_prompt,
    checkpointer=checkpointer,  # 启用对话记忆
    backend=filesystem_backend,  # 真实文件系统
    subagents=[research_subagent],  # 子代理
)


# 是否显示子智能体内部细节
SHOW_SUBAGENT_DETAILS = True

# 是否使用同步模式（invoke 而不是 astream）
USE_SYNC_MODE = False

# 流式模式: "tokens" = token级别流式, "nodes" = 节点级别流式
STREAM_MODE = "tokens"  # 改为 "nodes" 可以切换回节点级别


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
                    print(f"\n🔧 调用工具: {tc.get('name', 'unknown')} | 参数: {tc.get('args', {})}")
        
        elif msg_type == "tool":
            tool_name = getattr(msg, "name", "unknown")
            display_content = str(content)[:300] + "..." if len(str(content)) > 300 else content
            print(f"\n📦 [{tool_name}]: {display_content}")


# Token 级别流式输出
async def stream_tokens_response(user_input: str, config: dict):
    """Token 级别流式输出，每个 token 实时显示"""
    print("\n💬 AI: ", end="", flush=True)
    
    try:
        async for msg_chunk, metadata in agent.astream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode="messages",  # 关键：使用 messages 模式获取 token 级别流式
        ):
            # msg_chunk 是 AIMessageChunk 或其他消息类型
            msg_type = getattr(msg_chunk, "type", type(msg_chunk).__name__)
            
            if msg_type == "AIMessageChunk":
                content = getattr(msg_chunk, "content", "")
                if content:
                    print(content, end="", flush=True)  # 实时输出每个 token
                
                # 检查工具调用
                tool_calls = getattr(msg_chunk, "tool_calls", [])
                if tool_calls:
                    for tc in tool_calls:
                        if tc.get("name"):
                            print(f"\n\n🔧 调用工具: {tc.get('name')} | 参数: {tc.get('args', {})}", flush=True)
            
            elif msg_type == "tool":
                tool_name = getattr(msg_chunk, "name", "unknown")
                content = getattr(msg_chunk, "content", "")
                display_content = str(content)[:300] + "..." if len(str(content)) > 300 else content
                print(f"\n\n📦 [{tool_name}]: {display_content}", flush=True)
                print("\n💬 AI: ", end="", flush=True)  # 准备下一段 AI 输出
        
        print()  # 最后换行
    except Exception as e:
        error_str = str(e).lower()
        if "connection" in error_str or "timeout" in error_str:
            print(f"\n\n⚠️ 网络连接问题: {e}")
        else:
            print(f"\n\n❌ Token 流式错误: {e}")


# 流式输出单条消息（带子智能体细节）- 节点级别
async def stream_response(user_input: str, config: dict):
    """流式输出 agent 响应，支持显示子智能体内部细节"""
    
    try:
        if SHOW_SUBAGENT_DETAILS:
            # 使用 subgraphs=True 来显示子智能体内部细节
            async for chunk in agent.astream(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                stream_mode="updates",  # 使用 updates 模式获取增量更新
                subgraphs=True,  # 关键参数：显示子智能体内部执行
            ):
                # chunk 是一个元组: (namespace_path, update_dict)
                namespace, update = chunk
                
                # namespace 是元组，表示当前执行路径
                # 例如: () 表示主智能体, ('task:xxx',) 表示子智能体
                is_subagent = len(namespace) > 0
                prefix = "    🔹 [子代理] " if is_subagent else ""
                
                # 调试：打印所有 update 的键（可以注释掉）
                # print(f"  [DEBUG] namespace={namespace}, keys={list(update.keys())}")
                
                # 处理 model 节点的输出
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
                                    print(f"\n{prefix}🔧 调用工具: {tc.get('name', 'unknown')} | 参数: {tc.get('args', {})}")
                
                # 处理 tools 节点的输出
                if "tools" in update:
                    messages = update["tools"].get("messages", [])
                    for msg in messages:
                        tool_name = getattr(msg, "name", "unknown")
                        content = getattr(msg, "content", "")
                        display_content = str(content)[:300] + "..." if len(str(content)) > 300 else content
                        print(f"\n{prefix}📦 [{tool_name}]: {display_content}")
                
                # 处理其他可能的节点（如 __end__ 等）
                # 这些节点通常不需要特殊处理，但可以用于调试
        else:
            # 简化模式：不显示子智能体内部细节
            async for chunk in agent.astream(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                stream_mode="values"
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
                                print(f"\n🔧 调用工具: {tc.get('name', 'unknown')} | 参数: {tc.get('args', {})}")
                    
                    elif msg_type == "tool":
                        tool_name = getattr(msg, "name", "unknown")
                        display_content = str(content)[:300] + "..." if len(str(content)) > 300 else content
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
    # 声明全局变量（必须在使用前声明）
    global SHOW_SUBAGENT_DETAILS, USE_SYNC_MODE, STREAM_MODE, LANGSMITH_ENABLED
    
    # 创建唯一的对话线程 ID
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print("=" * 60)
    print("🤖 DeepAgents 多轮对话模式")
    print("=" * 60)
    print(f"📌 对话线程 ID: {thread_id}")
    print(f"📁 工作目录: {WORKSPACE_DIR}")
    if LANGSMITH_ENABLED:
        print(f"📊 LangSmith 追踪: 已启用 (项目: {LANGSMITH_PROJECT})")
    else:
        print("📊 LangSmith 追踪: 已禁用")
    print("💡 输入 'quit' 或 'exit' 退出对话")
    print("💡 输入 'new' 开始新对话")
    print("💡 输入 'toggle' 切换子代理细节显示")
    print("💡 输入 'sync' 切换同步/流式模式")
    print("💡 输入 'stream' 切换 token/节点 级别流式")
    print("💡 输入 'trace' 切换 LangSmith 追踪开关")
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
            
            if user_input.lower() == "trace":
                LANGSMITH_ENABLED = not LANGSMITH_ENABLED
                status = "开启" if LANGSMITH_ENABLED else "关闭"
                print(f"\n🔄 LangSmith 追踪已{status}")
                continue
            
            # 使用 tracing_context 包裹调用（不使用环境变量）
            try:
                with tracing_context(
                    client=langsmith_client,
                    project_name=LANGSMITH_PROJECT,
                    enabled=LANGSMITH_ENABLED,
                ):
                    # 根据模式选择调用方式
                    if USE_SYNC_MODE:
                        sync_response(user_input, config)
                    elif STREAM_MODE == "tokens":
                        await stream_tokens_response(user_input, config)
                    else:
                        await stream_response(user_input, config)
            except Exception as trace_error:
                # 如果是 LangSmith 追踪相关的连接错误，提示但继续运行
                error_str = str(trace_error).lower()
                if "connection" in error_str or "timeout" in error_str:
                    print("\n⚠️ LangSmith 追踪上传失败（网络问题），但对话已完成")
                    print("   提示：输入 'trace' 可禁用追踪")
                else:
                    raise  # 其他错误继续抛出
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(chat_loop())
