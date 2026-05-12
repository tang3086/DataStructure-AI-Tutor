import json
from environment import client

# 用于接收 main.py 传过来的索引，极大地提升检索速度
_GLOBAL_RAG = None

def set_rag_instance(instance):
    global _GLOBAL_RAG
    _GLOBAL_RAG = instance

def rag_search(query: str) -> str:
    """
    检索课程资料（极速版：取消中间过滤，直接返回原文）
    """
    if _GLOBAL_RAG is None:
        return "错误：知识库实例未初始化。"

    try:
        # 1. 直接检索 Top-3 最相关的原文片段
        # 备注：你可以把 top_k 设为 3，给主模型足够的素材
        results = _GLOBAL_RAG.search(query, top_k=3)
        
        # 2. 直接拼接原文，不做二次大模型处理
        context_list = []
        for i, (chunk, score) in enumerate(results):
            context_list.append(f"--- 资料片段 {i+1} (来源: {chunk.source}) ---\n{chunk.content}")
        
        raw_context = "\n\n".join(context_list)

        # 3. 直接返回给 Agent，让 Agent 自己在流式回答中去阅读和总结
        return f"找到以下教材相关原文：\n{raw_context}"

    except Exception as e:
        return f"检索异常: {str(e)}"
    
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