import json
from environment import client

# 用于接收 main.py 传过来的索引，极大地提升检索速度
_GLOBAL_RAG = None

def set_rag_instance(instance):
    global _GLOBAL_RAG
    _GLOBAL_RAG = instance

def rag_search(query: str) -> str:
    """
    检索课程资料并用大模型进行降噪过滤
    """
    if _GLOBAL_RAG is None:
        return "错误：知识库实例未初始化。"

    try:
        # 直接使用内存里的索引进行检索
        results = _GLOBAL_RAG.search(query, top_k=3)
        raw_context = "\n---\n".join([chunk.content for chunk, score in results])

        # 高质量过滤提示词，确保信息提取的严谨性
        filter_prompt = f"""你是一个数据结构课程助教。
学生的问题是：{query}
请从下面的教材文本中，提取出能直接回答该问题的核心知识点。只保留与问题直接相关的部分，删除无关的介绍、代码、目录、习题等。不要编造内容。

教材文本：
{raw_context}"""
        filter_response = client.chat.completions.create(
            model="qwen3.6-flash-2026-04-16",
            messages=[{"role": "user", "content": filter_prompt}],
            temperature=0.1,
            max_tokens=500
        )
        filtered_context = filter_response.choices[0].message.content

        return f"教材知识库检索结果：\n{filtered_context}"

    except Exception as e:
        return f"检索失败，请基于已有知识回答。错误信息：{str(e)}"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "当遇到数据结构、算法相关的概念解释、代码实现、复习资料请求时，必须调用此工具检索课程资料。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要检索的数据结构核心关键词，例如'单链表插入'、'快速排序'"}
                },
                "required": ["query"]
            }
        }
    }
]

tools_implementation = {
    "rag_search": rag_search
}