import gradio as gr
import json
from environment import client
from tools import TOOLS, tools_implementation, set_rag_instance
from rag_library import load_or_build_rag_library
from learning_log import LearningLogger 

# ================= 1. 初始化 =================
rag_instance = load_or_build_rag_library(client=client, verbose=True)
set_rag_instance(rag_instance)
MODEL_NAME = "qwen3.6-flash-2026-04-16"
logger = LearningLogger()  # 全局日志

# ================= 2. 系统提示词 =================
def build_web_system_prompt(language: str) -> str:
    return f"""
# 角色设定
你是顶尖大学的计算机系助教，也是学生专属的“数据结构与算法智能学伴”。
你性格耐心、专业、擅长启发式教学。

# 核心约束（最高优先级）
1. 【语言绝对锁定】：学生当前主攻的编程语言是 **{language}**。
   - 所有的代码示例、出题、变量命名习惯必须严格使用 {language}。
   - 在阅读检索到的教材资料时，如果原文只提供了 Java/C++ 等其他语言的代码，你必须在脑海中将其**转换**为 {language} 后再输出，绝不能直接照抄原文的其他语言代码。
2. 【边界控制】：只回答与“计算机、编程、数据结构、算法”相关的问题。

# 知识库使用规范 (RAG Rules)
- 当学生询问概念、原理或复习重点时，**必须**优先调用 `rag_search` 工具。

# 四大核心工作流
**模式一：概念讲解 (知识问答)**
- 📖 **核心定义**：一句话通俗解释。
- ⚙️ **底层原理**：逻辑拆解（必要时可带入生活中的比喻）。
- 💻 **代码示例**：提供一段精简、可运行的 {language} 代码。
- ⚠️ **避坑指南**：指出该知识点在考试或面试中的高频易错点。

**模式二：学情诊断 (Debug 与纠错)**
- 不要直接甩出正确答案。
- 像人类助教一样，精准指出是哪一行代码、什么逻辑出了问题。
- 使用**启发式提问**引导学生自己发现错误。

**模式三：个性化练习 (出题)**
- 结合刚学过的知识点，出一道符合期末考试难度的题目。
- 提供清晰的题目描述、示例输入/输出，并给出 {language} 的**代码骨架**（带 TODO 注释）。
- 【重要】：请在题目末尾提醒学生：“*💡 回答此题时，请在发言开头加上【提交答案】四个字。*”

**模式四：学习规划 (备考与突击)**
- 评估学生的剩余时间。
- 将庞大的知识体系拆解为“天”或“小时”级别的微小任务。
- 直接列出核心要背诵的算法框架和数据结构特性。
"""

# ================= 3. 核心辅助函数 =================
def detect_topic(user_msg: str, history: list) -> str:
    """
    【修复缺陷 1】: 动态从当前对话或历史上下文中提取真实的数据结构知识点标签
    """
    keywords = ["栈", "队列", "单链表", "双链表", "循环链表", "链表", "顺序表", "数组", "二叉树", "树", "图", "哈希", "散列", "排序", "查找", "时间复杂度"]
    
    # 1. 首先检查当前用户的发言
    for kw in keywords:
        if kw in user_msg:
            return kw
            
    # 2. 如果当前发言没明说，倒序扫描历史记录（寻找之前的出题或讨论上下文）
    for msg in reversed(history):
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in content])
        else:
            content = str(content)
            
        for kw in keywords:
            if kw in content:
                return kw
                
    return "通用基础"

def get_dashboard_data():
    """从 logger 中读取数据，格式化输出给前端看板"""
    logger._load()
    
    # 1. 组装学情画像
    weak_topics = logger.data.get("learning_profile", {}).get("weak_topics", [])
    strong_topics = logger.data.get("learning_profile", {}).get("strong_topics", [])
    
    profile_md = "### 📊 学习画像分析\n"
    profile_md += f"❌ **薄弱知识点**：{', '.join(weak_topics) if weak_topics else '暂无（继续保持！）'}\n\n"
    profile_md += f"✅ **熟练知识点**：{', '.join(strong_topics) if strong_topics else '暂无（多去练习吧！）'}\n"
    
    # 2. 【修复缺陷 2】: 将刷题记录改为按时间正序、且具备 HTML 原生可收放组件的形式
    exercises = logger.data.get("exercises", [])
    # 严格按照时间正序排列（最早的在前，最晚的在后）
    sorted_exercises = sorted(exercises, key=lambda x: x.get("timestamp", ""))
    
    exercise_md = "### ✏️ 刷题历史记录\n"
    if not sorted_exercises:
        exercise_md += "*暂无做题记录，可以让学伴给你出一道题试试！*"
    else:
        for item in sorted_exercises:
            time_str = item['timestamp'][:16].replace('T', ' ')
            topic_str = item.get('topic') or '未知考点'
            
            # 使用 HTML <details> 和 <summary> 实现精美的可折叠看板行
            exercise_md += f"<details style='margin-bottom: 8px; border: 1px solid #e0e0e0; padding: 6px; border-radius: 4px;'>\n"
            exercise_md += f"<summary style='cursor: pointer; font-weight: bold; color: #4F46E5;'>📅 {time_str} | 💡 {topic_str}</summary>\n\n"
            exercise_md += f"**🧩 题目概要**：{item['question'][:80]}...\n\n"
            exercise_md += f"**💬 我的解答**：\n```text\n{item['user_answer']}\n```\n\n"
            exercise_md += f"**💡 助教点评**：\n{item['feedback']}\n"
            exercise_md += f"</details>\n"
            
    return profile_md, exercise_md

def user_action(user_message, history):
    if not user_message.strip():
        return gr.update(), history
    return "", history + [{"role": "user", "content": user_message}]

def bot_action(history, selected_language):
    if not history or history[-1]["role"] != "user":
        yield history, *get_dashboard_data()
        return

    sys_prompt = build_web_system_prompt(selected_language)

    llm_messages = [{"role": "system", "content": sys_prompt}]
    for msg in history[:-1]:
        llm_messages.append({"role": msg["role"], "content": msg["content"]})
    llm_messages.append({"role": "user", "content": history[-1]["content"]})

    history.append({"role": "assistant", "content": ""})
    yield history, *get_dashboard_data()

    # 解包脱壳，确保类型安全
    raw_user_msg = history[-2]["content"]
    if isinstance(raw_user_msg, list):
        user_msg = "".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in raw_user_msg])
    else:
        user_msg = str(raw_user_msg)

    full_content = ""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=llm_messages,
            tools=TOOLS,
            stream=True
        )

        tool_calls_chunks = []
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                full_content += delta.content
                history[-1]["content"] = full_content
                yield history, *get_dashboard_data()
            
            if delta.tool_calls:
                for tc_chunk in delta.tool_calls:
                    if len(tool_calls_chunks) <= tc_chunk.index:
                        tool_calls_chunks.append({
                            "id": tc_chunk.id, "type": "function",
                            "function": {"name": "", "arguments": ""}
                        })
                    if tc_chunk.function.name:
                        tool_calls_chunks[tc_chunk.index]["function"]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        tool_calls_chunks[tc_chunk.index]["function"]["arguments"] += tc_chunk.function.arguments

        if tool_calls_chunks:
            history[-1]["content"] = full_content + "\n\n*(🔍 正在翻阅教材资料...)*"
            yield history, *get_dashboard_data()
            
            llm_messages.append({"role": "assistant", "content": full_content, "tool_calls": tool_calls_chunks})
            
            for tc in tool_calls_chunks:
                args = json.loads(tc["function"]["arguments"] or "{}")
                result = tools_implementation["rag_search"](**args)
                llm_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            
            final_res = client.chat.completions.create(
                model=MODEL_NAME,
                messages=llm_messages,
                stream=True
            )
            for chunk in final_res:
                if chunk.choices[0].delta.content:
                    full_content += chunk.choices[0].delta.content
                    history[-1]["content"] = full_content
                    yield history, *get_dashboard_data()
    
    finally:
        raw_assistant_msg = history[-1]["content"]
        if isinstance(raw_assistant_msg, list):
            assistant_msg = "".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in raw_assistant_msg])
        else:
            assistant_msg = str(raw_assistant_msg)
        
        # 触发智能分析与精准日志沉淀
        if user_msg.startswith("【提交答案】"):
            # 智能提取真实的课程核心知识点
            detected_topic = detect_topic(user_msg, history[:-1])
            
            logger.log_exercise(
                question="智能测试题（详见聊天上下文）", 
                user_answer=user_msg, 
                feedback=assistant_msg, 
                topic=detected_topic
            )
            # 如果判定到错误，将其拉入对应的薄弱点库
            if any(w in assistant_msg for w in ["错", "错误", "bug", "瑕疵", "不够准确", "不完全正确"]):
                logger.log_error(topic=detected_topic, description=user_msg[:100])
        else:
            logger.log_conversation(user_msg, assistant_msg)
            
        yield history, *get_dashboard_data()


# ================= 4. 构建界面 =================
init_profile, init_exercise = get_dashboard_data()

with gr.Blocks(title="数据结构智能学伴") as demo:
    gr.Markdown("# 🎓 数据结构智能学伴")

    with gr.Row():
        # 左栏：对话区
        with gr.Column(scale=7):
            language_selector = gr.Radio(
                choices=["C语言", "Python"],
                label="请选择主攻编程语言",
                value="C语言"
            )

            chatbot = gr.Chatbot(label="聊天历史", height=550)

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="在此输入疑问，或以【提交答案】开头回复学伴的出题...",
                    show_label=False,
                    scale=9
                )
                submit_btn = gr.Button("发送", variant="primary", scale=1)
                
        # 右栏：学情看板区
        with gr.Column(scale=3):
            gr.Markdown("## 📋 我的学情看板")
            
            profile_box = gr.Markdown(value=init_profile, line_breaks=True)
            gr.Markdown("---")
            exercise_box = gr.Markdown(value=init_exercise, line_breaks=True)
            
            refresh_btn = gr.Button("🔄 手动刷新看板数据", size="sm")

    # ================= 5. 事件流绑定 =================
    msg.submit(
        user_action, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False
    ).then(
        bot_action, 
        inputs=[chatbot, language_selector], 
        outputs=[chatbot, profile_box, exercise_box]
    )
    
    submit_btn.click(
        user_action, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False
    ).then(
        bot_action, 
        inputs=[chatbot, language_selector], 
        outputs=[chatbot, profile_box, exercise_box]
    )
    
    refresh_btn.click(
        get_dashboard_data, 
        inputs=[], 
        outputs=[profile_box, exercise_box]
    )

if __name__ == "__main__":
    demo.launch(
        inbrowser=True,
        theme=gr.themes.Soft(),
        css="footer {visibility: hidden}"
    )