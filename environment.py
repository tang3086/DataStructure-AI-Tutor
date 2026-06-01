import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 在这里读取环境变量，确保在运行时可以访问到它们
load_dotenv(override=True)

# 这段代码检查环境变量是否存在，并创建一个 OpenAI 客户端实例。
# 如果缺少必要的环境变量，它会抛出一个错误，提示用户设置它们。
# 大部分的模型厂商都适配的 openai 的接口规范，所以你可以通过修改 BASE_URL 和 API_KEY 来切换不同厂商的模型。
if os.getenv("BASE_URL") and os.getenv("API_KEY"):
    client = OpenAI(
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
    )
else:
    raise EnvironmentError("Please set API_KEY and BASE_URL in your environment variables.")