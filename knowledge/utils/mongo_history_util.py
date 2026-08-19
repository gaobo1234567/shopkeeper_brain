import os
import logging
from typing import List, Dict, Any
from datetime import datetime

from pymongo import MongoClient, ASCENDING, DESCENDING
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()


class HistoryMongoTool:
    """
    MongoDB 历史对话记录读写工具。

    当前包含两个 collection：
    1. chat_message：
       保存真实的用户/助手聊天记录。

    2. chat_session_state：
       保存跨轮次的会话业务状态。
       目前主要用于“商品歧义澄清”流程。
    """

    def __init__(self):
        """
        初始化 MongoDB 连接。
        """
        try:
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")

            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]

            # -----------------------------
            # 1. 聊天消息 collection
            # -----------------------------
            self.chat_message = self.db["chat_message"]

            # session_id + ts 索引，用于快速获取某个会话的历史消息
            self.chat_message.create_index(
                [("session_id", ASCENDING), ("ts", DESCENDING)]
            )

            # -----------------------------
            # 2. 会话状态 collection
            # -----------------------------
            self.chat_session_state = self.db["chat_session_state"]

            # 一个 session 只能存在一条当前业务状态
            self.chat_session_state.create_index(
                [("session_id", ASCENDING)],
                unique=True
            )

            logging.info(
                f"Successfully connected to MongoDB: {self.db_name}"
            )

        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            raise


def clear_history(session_id: str) -> int:
    """
    清空指定会话的历史记录。

    除了删除 chat_message，
    同时删除该 session 对应的 pending clarification 状态，
    防止用户清空对话后仍然残留“等待商品选择”的状态。

    :param session_id: 会话 ID
    :return: 被删除的聊天消息数量
    """

    mongo_tool = get_history_mongo_tool()

    try:
        # 1. 删除聊天历史
        result = mongo_tool.chat_message.delete_many(
            {"session_id": session_id}
        )

        # 2. 删除会话业务状态
        mongo_tool.chat_session_state.delete_one(
            {"session_id": session_id}
        )

        logging.info(
            f"Deleted {result.deleted_count} messages "
            f"and session state for session {session_id}"
        )

        return result.deleted_count

    except Exception as e:
        logging.error(
            f"Error clearing history for session {session_id}: {e}"
        )
        return 0


def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: List[str] = None,
        message_id: str = None
) -> str:
    """
    写入一条会话记录。

    :param message_id: 主键；存在时更新，不存在时新增
    :param rewritten_query: 重写后的查询
    :param session_id: 会话 ID
    :param role: user / assistant
    :param text: 消息文本
    :param item_names: 当前消息关联的商品名称
    :return: ObjectId 字符串
    """

    ts = datetime.now().timestamp()

    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query,
        "item_names": item_names,
        "ts": ts
    }

    mongo_tool = get_history_mongo_tool()

    # 有 message_id：更新原记录
    if message_id:
        mongo_tool.chat_message.update_one(
            {"_id": ObjectId(message_id)},
            {"$set": document}
        )
        return message_id

    # 没有 message_id：插入新记录
    result = mongo_tool.chat_message.insert_one(document)

    return str(result.inserted_id)


def update_message_item_names(
        ids: List[str],
        item_names: List[str]
) -> int:
    """
    批量更新历史消息中的 item_names。

    仅更新：
    - item_names 字段不存在
    - item_names == []
    - item_names == None

    的记录。

    :param ids: MongoDB ObjectId 字符串列表
    :param item_names: 要回填的商品名称
    :return: 实际修改数量
    """

    mongo_tool = get_history_mongo_tool()

    try:
        object_ids = [ObjectId(i) for i in ids]

        result = mongo_tool.chat_message.update_many(
            {
                "_id": {"$in": object_ids},
                "$or": [
                    {"item_names": {"$exists": False}},
                    {"item_names": []},
                    {"item_names": None}
                ]
            },
            {
                "$set": {
                    "item_names": item_names
                }
            }
        )

        logging.info(
            f"Updated {result.modified_count} records "
            f"to item_names: {item_names}"
        )

        return result.modified_count

    except Exception as e:
        logging.error(
            f"Error updating history item_names: {e}"
        )
        return 0


def get_recent_messages(
        session_id: str,
        limit: int = 10
) -> List[Dict[str, Any]]:
    """
    查询最近 N 条对话记录。

    注意：
    数据库查询时：
        ts DESCENDING
    先拿到最新的 N 条。

    返回给 LLM 前：
        reverse()
    恢复为“旧 -> 新”的自然对话顺序。

    例如数据库有：
        1,2,3,...,100

    limit=10 时：
        数据库先查 100~91
        再 reverse
        返回 91~100

    :param session_id: 会话 ID
    :param limit: 最近消息数量
    :return: 按时间正序排列的最近 N 条消息
    """

    mongo_tool = get_history_mongo_tool()

    try:
        query = {
            "session_id": session_id
        }

        cursor = (
            mongo_tool.chat_message
            .find(query)
            .sort("ts", DESCENDING)
            .limit(limit)
        )

        messages = list(cursor)

        # 数据库查出来是 新 -> 旧
        # LLM需要 旧 -> 新
        messages.reverse()

        return messages

    except Exception as e:
        logging.error(
            f"Error getting recent messages: {e}"
        )
        return []


def save_pending_clarification(
        session_id: str,
        pending_options: List[str],
        pending_query: str
) -> None:
    """
    保存当前会话的“待用户选择商品”状态。

    典型场景：

    用户：
        华为擎云怎么恢复系统？

    系统发现：
        G540
        G740

    此时保存：

        status = waiting_item_selection
        pending_options = [G540, G740]
        pending_query = 原始问题

    后续用户回复：
        第二个

    就可以基于该状态继续处理，而不是重新从零开始理解。

    :param session_id: 会话 ID
    :param pending_options: 待用户选择的商品名称
    :param pending_query: 发生歧义时原始业务问题
    """

    mongo_tool = get_history_mongo_tool()

    document = {
        "session_id": session_id,
        "status": "waiting_item_selection",
        "pending_options": pending_options,
        "pending_query": pending_query,
        "updated_at": datetime.now().timestamp()
    }

    try:
        mongo_tool.chat_session_state.update_one(
            {
                "session_id": session_id
            },
            {
                "$set": document
            },
            upsert=True
        )

        logging.info(
            f"Saved pending clarification: "
            f"session={session_id}, "
            f"options={pending_options}"
        )

    except Exception as e:
        logging.error(
            f"Error saving pending clarification "
            f"for session {session_id}: {e}"
        )


def get_pending_clarification(
        session_id: str
) -> Dict[str, Any]:
    """
    获取当前会话的商品澄清状态。

    如果当前 session 正在等待用户选择商品，
    返回类似：

    {
        "session_id": "...",
        "status": "waiting_item_selection",
        "pending_options": [...],
        "pending_query": "...",
        "updated_at": ...
    }

    如果不存在 pending 状态，则返回 {}。

    :param session_id: 会话 ID
    :return: pending state
    """

    mongo_tool = get_history_mongo_tool()

    try:
        result = mongo_tool.chat_session_state.find_one(
            {
                "session_id": session_id,
                "status": "waiting_item_selection"
            }
        )

        return result or {}

    except Exception as e:
        logging.error(
            f"Error getting pending clarification "
            f"for session {session_id}: {e}"
        )
        return {}


def clear_pending_clarification(
        session_id: str
) -> int:
    """
    清除当前会话的商品澄清状态。

    用户成功完成商品选择后必须调用，
    否则下一轮正常问题可能仍然被误判为：
        “用户正在回答上一次商品选择”。

    :param session_id: 会话 ID
    :return: 删除数量
    """

    mongo_tool = get_history_mongo_tool()

    try:
        result = mongo_tool.chat_session_state.delete_one(
            {
                "session_id": session_id
            }
        )

        logging.info(
            f"Cleared pending clarification: "
            f"session={session_id}, "
            f"deleted={result.deleted_count}"
        )

        return result.deleted_count

    except Exception as e:
        logging.error(
            f"Error clearing pending clarification "
            f"for session {session_id}: {e}"
        )
        return 0


# ============================================================
# MongoDB 单例
# ============================================================

_history_mongo_tool = None


def get_history_mongo_tool() -> HistoryMongoTool:
    """
    获取 MongoDB 工具单例。
    """

    global _history_mongo_tool

    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()

    return _history_mongo_tool


# 模块加载时尝试初始化
try:
    _history_mongo_tool = HistoryMongoTool()

except Exception as e:
    logging.warning(
        f"Could not initialize HistoryMongoTool on module load: {e}"
    )