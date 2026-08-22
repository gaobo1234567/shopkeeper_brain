import logging
from typing import Optional

import redis

logger = logging.getLogger(__name__)

# 全局 Redis 客户端
_redis_client: Optional[redis.Redis] = None


def get_redis_client() -> Optional[redis.Redis]:
    """
    获取 Redis 客户端。

    使用单例方式复用连接池，避免每次请求都重新创建 Redis 连接。
    Redis 不可用时返回 None，不影响主业务流程。
    """
    global _redis_client

    # 已经初始化过，直接复用
    if _redis_client is not None:
        return _redis_client

    try:
        client = redis.Redis(
            host="localhost",
            port=6379,
            db=0,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )

        # 主动测试 Redis 是否可用
        client.ping()

        _redis_client = client

        logger.info("Redis 客户端初始化成功")

        return _redis_client

    except Exception as e:
        logger.error(f"Redis 客户端初始化失败: {e}")
        return None
if __name__ == "__main__":
    client = get_redis_client()

    if client is None:
        print("Redis 连接失败")
    else:
        client.set("test_key", "hello_redis", ex=60)

        value = client.get("test_key")

        print("Redis 连接成功")
        print("读取结果:", value)