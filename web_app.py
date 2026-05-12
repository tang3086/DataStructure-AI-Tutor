import gradio as gr
import json
from environment import client
from tools import TOOLS, tools_implementation, set_rag_instance
from rag_library import load_or_build_rag_library

# ================= 1. 初始化 =================
rag_instance = load_or_build_rag_library(client=client, verbose=True)
set_rag_instance(rag_instance)
MODEL_NAME = "qwen3.6-flash-2026-04-16"

def predict(user_message, history):
    current_language = "C语言" 
    sys_prompt = f"""
    你是一个专业的数据结构学伴。当前语言锁定为：{current_language}。
    请严格遵守：1.知识问答优先检索教材。2.出题和代码解析必须使用{current_language}。
    """

    # 1. 构造发给大模型的 messages
    # Gradio 6.0 的 history 直接就是 [{"role": "user", "content": "..."}, ...] 格式！
    llm_messages = [{"role": "system", "content": sys_prompt}]
    for msg in history:
        llm_messages.append({"role": msg["role"], "content": msg["content"]})
    llm_messages.append({"role": "user", "content": user_message})

    # 2. 准备 UI 上的聊天记录
    # 将用户的新问题追加到界面历史中
    history.append({"role": "user", "content": user_message})
    # 预留一个机器人的空位，用于接收流式打字数据
    history.append({"role": "assistant", "content": ""})

    # 第一轮：发起大模型请求
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=llm_messages,
        tools=TOOLS,
        stream=True
    )

    full_content = ""
    tool_calls_chunks = []

    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            full_content += delta.content
            # 更新 history 中最后一个元素（即机器人的回复），并推送到前端
            history[-1]["content"] = full_content
            yield history
        
        # 收集碎片化的工具调用 (RAG)
        if delta.tool_calls:
            for tc_chunk in delta.tool_calls:
                if len(tool_calls_chunks) <= tc_chunk.index:
                    tool_calls_chunks.append({"id": tc_chunk.id, "type": "function", "function": {"name": "", "arguments": ""}})
                if tc_chunk.function.name:
                    tool_calls_chunks[tc_chunk.index]["function"]["name"] += tc_chunk.function.name
                if tc_chunk.function.arguments:
                    tool_calls_chunks[tc_chunk.index]["function"]["arguments"] += tc_chunk.function.arguments

    # 如果触发了工具调用
    if tool_calls_chunks:
        # UI 提示正在检索
        history[-1]["content"] = full_content + "\n\n*(🔍 正在翻阅教材资料...)*"
        yield history
        
        # 把工具意图压入对话记录
        llm_messages.append({"role": "assistant", "content": full_content, "tool_calls": tool_calls_chunks})
        
        # 真正执行工具
        for tc in tool_calls_chunks:
            args = json.loads(tc["function"]["arguments"] or "{}")
            result = tools_implementation["rag_search"](**args)
            llm_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        
        # 第二轮：带着搜到的资料再次请求
        final_res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=llm_messages,
            stream=True
        )
        for chunk in final_res:
            if chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
                history[-1]["content"] = full_content
                yield history

# ================= 2. 界面布局 (Gradio 6.0 极简风) =================
# 删除了所有被 Gradio 6.0 废弃的组件参数
with gr.Blocks() as demo:
    gr.Markdown("# 🎓 数据结构智能学伴 (Web 版)")
    
    chatbot = gr.Chatbot(
        label="聊天历史",
        height=600
        # 移除了 type 和 show_copy_button，直接使用 6.0 默认的最优配置
    )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="在此输入代码或疑问 (按 Enter 发送)...",
            show_label=False,
            scale=9
        )
        submit_btn = gr.Button("发送", variant="primary", scale=1)

    # 绑定发送事件
    msg.submit(predict, [msg, chatbot], [chatbot])
    msg.submit(lambda: "", None, msg) # 发送后清空输入框
    submit_btn.click(predict, [msg, chatbot], [chatbot])
    submit_btn.click(lambda: "", None, msg)

if __name__ == "__main__":
    # 根据官方规范，把 theme 和 css 放到了 launch 里面
    demo.launch(
        inbrowser=True, 
        theme=gr.themes.Soft(), 
        css="footer {visibility: hidden}"
    )