import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# 用来保存已经创建过的客户端对象
cache_llm_client = {}


def get_llm_client(
    mode_name: str = None,
    temperature: float = 0.0,
    response_format: bool = False
):
    """
    返回 LLM 客户端对象。

    缓存：
        value: ChatOpenAI client
        key: 不同模型 + 不同响应格式
    """

    # 1. 获取模型配置
    model_name = mode_name or os.getenv("ITEM_MODEL")
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")

    # 2. 获取 LLM Timeout
    llm_timeout_seconds = float(
        os.getenv("LLM_TIMEOUT_SECONDS", "10")
    )

    # 3. 客户端缓存 key
    cache_key = (
        model_name,
        response_format
    )

    # 4. 缓存命中
    if cache_key in cache_llm_client:
        return cache_llm_client[cache_key]

    # 5. 返回内容格式
    model_kwargs = {}

    if response_format:
        model_kwargs["response_format"] = {
            "type": "json_object"
        }

    try:
        # 6. 创建 LLM 客户端
        client = ChatOpenAI(
            model_name=model_name,
            openai_api_key=api_key,
            openai_api_base=api_base,
            temperature=temperature,
            # LLM 请求最多允许执行多少秒
            timeout=llm_timeout_seconds,
            model_kwargs=model_kwargs
        )

        # 7. 缓存客户端
        cache_llm_client[cache_key] = client

        # 8. 返回客户端
        return client

    except Exception as e:
        logger.error(
            "LLM客户端创建失败, 原因: %s",
            str(e)
        )
        return None


if __name__ == "__main__":
    llm_client = get_llm_client()
    import json
    ai_message = llm_client.invoke('您好，请给我讲一个笑话，返回json格式：{"key":"value"}')
    print(ai_message.content)
    print(type(ai_message.content))
    json_object = json.loads(ai_message.content)
    print(type(json_object))