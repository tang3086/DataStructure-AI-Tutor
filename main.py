import json
import os
import sys
from environment import client
from tools import TOOLS, tools_implementation, set_rag_instance
from rag_library import load_or_build_rag_library

# ================= 配置区 =================
# 全局唯一 RAG 实例，启动时仅加载一次（提速核心）
rag_instance = load_or_build_rag_library(client=client, verbose=True)
# 将实例注入给工具箱
set_rag_instance(rag_instance)

MODEL_NAME = "qwen3.6-flash-2026-04-16"
# ==========================================

def build_rag_system_prompt(current_language: str) -> str:
    return f"""
# 角色设定
你是顶尖大学的计算机系助教，也是学生专属的“数据结构与算法智能学伴”。
你性格耐心、专业、擅长启发式教学。你不仅提供答案，更注重引导学生独立思考。

# 核心约束（最高优先级）
1. 【语言绝对锁定】：学生当前主攻的编程语言是 **{current_language}**。
   - 所有的代码示例、出题、变量命名习惯必须严格使用 {current_language}。
   - 在阅读检索到的教材资料时，如果原文只提供了 Java/C++ 等其他语言的代码，你必须在脑海中将其**转换**为 {current_language} 后再输出，绝不能直接照抄原文的其他语言代码。
2. 【边界控制】：只回答与“计算机、编程、数据结构、算法”相关的问题。如果学生聊起无关话题，请礼貌地将其引导回学习上。

# 知识库使用规范 (RAG Rules)
- 当学生询问概念、原理或复习重点时，**必须**优先调用 `rag_search` 工具。
- 你接收到的检索片段可能是原始的、带有杂讯的文本。你需要具备“大海捞针”的能力，自行过滤掉无关信息，只提取核心逻辑。
- 如果教材原文的内容足以回答，优先基于原文回答；如果原文未提及，你可以使用自身知识库补充，但在语气上无需刻意强调来源。

# 四大核心工作流
请根据学生的输入意图，灵活采用以下模式之一进行回复：

**模式一：概念讲解 (知识问答)**
当学生问“什么是...”、“怎么理解...”时，请严格采用以下结构排版：
- 📖 **核心定义**：一句话通俗解释。
- ⚙️ **底层原理**：逻辑拆解（必要时可带入生活中的比喻）。
- 💻 **代码示例**：提供一段精简、可运行的 {current_language} 代码。
- ⚠️ **避坑指南**：指出该知识点在考试或面试中的高频易错点。

**模式二：学情诊断 (Debug 与纠错)**
当学生提供自己的代码或思路时：
- 不要直接甩出正确答案。
- 像人类助教一样，精准指出是哪一行代码、什么逻辑出了问题（如：数组越界、指针断裂）。
- 使用**启发式提问**（如：“如果你输入的是空链表，你的第3行代码会发生什么？”），引导学生自己发现错误。

**模式三：个性化练习 (出题)**
当学生要求练习时：
- 结合刚学过的知识点，出一道符合期末考试难度的题目。
- 提供清晰的题目描述、示例输入/输出，并给出 {current_language} 的**代码骨架**（带 TODO 注释）让学生填空或补全。
- 等待学生提交代码后，再进行打分和点评。

**模式四：学习规划 (备考与突击)**
当学生焦虑或需要复习建议时：
- 评估学生的剩余时间。
- 将庞大的知识体系拆解为“天”或“小时”级别的微小任务。
- 直接列出核心要背诵的算法框架和数据结构特性。
"""

def run_agent_stream(messages, sys_prompt):
    """
    备注：这是支持流式输出的核心函数。
    它会实时打印文字，并在后台累积工具调用指令。
    """
    while True:
        # 开启 stream=True
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": sys_prompt}] + messages,
            tools=TOOLS,
            stream=True 
        )

        full_content = ""
        tool_calls_chunks = [] # 用于收集碎片化的工具指令

        print(f"\033[32m学伴 >>\033[0m ", end="")
        sys.stdout.flush()

        for chunk in stream:
            delta = chunk.choices[0].delta
            
            # 1. 处理普通文字内容
            if delta.content:
                content = delta.content
                full_content += content
                print(content, end="") # 实时打印，实现打字机效果
                sys.stdout.flush()

            # 2. 处理工具调用碎片
            if delta.tool_calls:
                for tc_chunk in delta.tool_calls:
                    # 如果是该索引的第一个碎片，初始化它
                    if len(tool_calls_chunks) <= tc_chunk.index:
                        tool_calls_chunks.append({
                            "id": tc_chunk.id,
                            "type": "function",
                            "function": {"name": "", "arguments": ""}
                        })
                    
                    # 合并碎片内容
                    if tc_chunk.function.name:
                        tool_calls_chunks[tc_chunk.index]["function"]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        tool_calls_chunks[tc_chunk.index]["function"]["arguments"] += tc_chunk.function.arguments

        print() # 回车

        # 将完整的助手回复存入历史
        assistant_msg = {"role": "assistant", "content": full_content}
        if tool_calls_chunks:
            # 格式化为 API 需要的格式
            formatted_calls = []
            for tc in tool_calls_chunks:
                formatted_calls.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": tc["function"]
                })
            assistant_msg["tool_calls"] = formatted_calls
        
        messages.append(assistant_msg)

        # 如果没有工具调用，说明本轮对话结束
        if not tool_calls_chunks:
            return

        # 执行工具调用
        for tc in formatted_calls:
            tool_name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"] or "{}")
            tool_fn = tools_implementation.get(tool_name)
            
            print(f"\033[33m[正在检索教材资料...] 关键词: {args.get('query')}\033[0m")
            result = tool_fn(**args)
            
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })
        print("\033[33m[资料阅读完毕，学伴正在总结...]\033[0m")

if __name__ == "__main__":
    print("\033[35m=======================================\033[0m")
    print("\033[35m  欢迎使用数据结构智能学伴！\033[0m")
    print("\033[35m=======================================\033[0m")
    
    # 语言选择环节
    while True:
        lang_choice = input("\033[33m请选择你要主攻的编程语言 输入 1 代表 C语言，2 代表 Python >> \033[0m").strip()
        if lang_choice == "1":
            current_language = "C语言"
            break
        elif lang_choice == "2":
            current_language = "Python"
            break
        else:
            print("\033[31m输入无效，请输入 1 或 2。\033[0m")

    print(f"\n\033[32m[系统设置] 已将学伴辅导语言锁定为 {current_language}\033[0m")
    print("\033[35m你可以问我概念、让我出题，或者帮你做复习计划。\033[0m\n")

    history = []
    while True:
        print("\033[36m你 >> 请输入内容。如果是多行代码，请在最后单起一行输入 EOF 并回车提交：\033[0m")
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "EOF":
                    break
                lines.append(line)
            except (EOFError, KeyboardInterrupt):
                exit()
                
        query = "\n".join(lines).strip()
        if not query or query.lower() in ("q", "exit"):
            break

        history.append({"role": "user", "content": query})
        
        final_system_prompt = build_rag_system_prompt(current_language)
        print("\033[33m[学伴正在思考中，请稍候...]\033[0m")
        
        # 还原原有的防崩溃报错机制
        try:
            answer = run_agent_stream(history, final_system_prompt)
            print(f"\033[32m学伴 >>\033[0m {answer}\n")
        except Exception as e:
            print(f"\033[31m哎呀，程序遇到了一点小网络问题：{str(e)}\033[0m\n")