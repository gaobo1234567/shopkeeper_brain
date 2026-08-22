import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import hashlib
from knowledge.utils.redis_util import get_redis_client
from typing import List, Tuple, Union,Any,Dict
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError

from knowledge.utils.llm_client_util import get_llm_client
from knowledge.prompts.query.query_prompt import USER_HYDE_PROMPT_TEMPLATE
from knowledge.utils.milvus_util import get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.bge_m3_embedding_util import generate_hybrid_embeddings, get_beg_m3_embedding_model

HYDE_CACHE_TTL_SECONDS = 24 * 60 * 60
class HyDeSearchNode(BaseNode):
    name = "hyde_search_node"

    def _build_hyde_cache_key(
            self,
            validated_query: str,
            validate_item_names: List[str]
    ) -> str:
        """
        根据问题和商品名生成稳定的 Redis 缓存键。
        """
        normalized_query = " ".join(validated_query.split()).strip()

        normalized_item_names = sorted(
            name.strip()
            for name in validate_item_names
            if name and name.strip()
        )

        raw_key = (
                normalized_query
                + "|"
                + "|".join(normalized_item_names)
        )

        digest = hashlib.sha256(
            raw_key.encode("utf-8")
        ).hexdigest()

        return f"hyde:{digest}"

    def _get_cached_hy_document(
            self,
            cache_key: str
    ) -> str:
        """
        从 Redis 获取 HyDE 假设文档。

        Redis 异常时直接视为缓存未命中，
        不影响主查询流程。
        """
        try:
            redis_client = get_redis_client()

            if redis_client is None:
                return ""

            cached_document = redis_client.get(cache_key)

            if cached_document:
                self.logger.info(
                    "HyDE 缓存命中 key=%s",
                    cache_key
                )
                return cached_document

            self.logger.info(
                "HyDE 缓存未命中 key=%s",
                cache_key
            )

            return ""

        except Exception as e:
            self.logger.warning(
                "读取 HyDE 缓存失败，继续调用 LLM: %s",
                e
            )
            return ""

    def _set_cached_hy_document(
            self,
            cache_key: str,
            hy_document: str
    ) -> None:
        """
        将 HyDE 假设文档写入 Redis。
        """
        if not hy_document:
            return

        try:
            redis_client = get_redis_client()

            if redis_client is None:
                return

            redis_client.set(
                cache_key,
                hy_document,
                ex=HYDE_CACHE_TTL_SECONDS
            )

            self.logger.info(
                "HyDE 缓存写入成功 key=%s ttl=%s",
                cache_key,
                HYDE_CACHE_TTL_SECONDS
            )

        except Exception as e:
            self.logger.warning(
                "写入 HyDE 缓存失败，不影响主流程: %s",
                e
            )
    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:

        try:
            # 1. 参数校验
            validated_query, validate_item_names = self._validate_query_inputs(state)

            # 2. 生成假设性文档
            hy_document = self._generate_hy_document(
                validated_query,
                validate_item_names
            )

            # HyDE 文档生成失败：
            # 不影响其他 Retriever，当前 HyDE 分支直接降级为空结果
            if not hy_document:
                self.logger.warning(
                    "HyDE 假设性文档生成失败，当前分支降级为空结果"
                )
                return {"hyde_embedding_chunks": []}

            # 3. 获取嵌入模型 & Milvus 客户端
            embedding_model = get_beg_m3_embedding_model()
            milvus_client = get_milvus_client()

            if not embedding_model or not milvus_client:
                self.logger.warning(
                    "HyDE Embedding 或 Milvus 客户端初始化失败，当前分支降级为空结果"
                )
                return {"hyde_embedding_chunks": []}

            # 4. 假设性文档嵌入
            embedding_document = f"{validated_query}\n{hy_document}"

            embedding_result = generate_hybrid_embeddings(
                embedding_model,
                embedding_documents=[embedding_document]
            )

            if not embedding_result:
                self.logger.warning(
                    "HyDE Embedding 生成失败，当前分支降级为空结果"
                )
                return {"hyde_embedding_chunks": []}

            # 5. 获取 item_name 的过滤表达式
            item_name_filtered_expr = self._item_name_filte_expr(
                validate_item_names
            )

            # 6. 创建混合搜索请求
            hybrid_search_requests = create_hybrid_search_requests(
                dense_vector=embedding_result["dense"][0],
                sparse_vector=embedding_result["sparse"][0],
                expr=item_name_filtered_expr
            )

            # 7. 执行 Milvus 混合搜索
            reps = execute_hybrid_search_query(
                milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=hybrid_search_requests,
                norm_score=True,
                output_fields=[
                    "chunk_id",
                    "content",
                    "item_name"
                ]
            )

            if not reps or not reps[0]:
                self.logger.warning(
                    "HyDE 未检索到结果，当前分支返回空结果"
                )
                return {"hyde_embedding_chunks": []}

            # 8. 正常返回 HyDE 检索结果
            return {
                "hyde_embedding_chunks": reps[0]
            }

        except StateFieldError:
            # rewritten_query / item_names 等 State 契约错误，
            # 属于程序逻辑问题，不能悄悄降级
            raise

        except Exception as e:
            # LLM / Embedding / Milvus 等运行时依赖异常：
            # 只让 HyDE 当前分支失败，不影响 Hybrid / KG
            self.logger.exception(
                "HyDE Retriever 运行异常，当前分支降级为空结果: %s",
                e
            )

            return {
                "hyde_embedding_chunks": []
            }

    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:

        # 1. 获取state的rewritten_query
        rewritten_query = state.get('rewritten_query', "")

        # 2. 获取state的item_names
        item_names = state.get('item_names', "")

        # 3. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        # 4. 返回
        return rewritten_query, item_names

    def _generate_hy_document(
            self,
            validated_query: str,
            validate_item_names: List[str]
    ) -> str:

        # 1. 构造缓存键
        cache_key = self._build_hyde_cache_key(
            validated_query,
            validate_item_names
        )

        # 2. 先查 Redis
        cached_document = self._get_cached_hy_document(
            cache_key
        )

        if cached_document:
            return cached_document

        # 3. 缓存未命中，再调用 LLM
        llm_client = get_llm_client()

        if llm_client is None:
            return ""

        user_prompt = USER_HYDE_PROMPT_TEMPLATE.format(
            item_hint=validate_item_names,
            rewritten_query=validated_query
        )

        system_prompt = (
            f"您是一位{validate_item_names}的技术文档领域的专家，"
            "主要擅长编写技术文档、操作手册、文档规格说明"
        )

        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            llm_response_content = getattr(
                llm_response,
                "content",
                ""
            ).strip()

            if not llm_response_content:
                return ""

            # 4. LLM 成功后写入 Redis
            self._set_cached_hy_document(
                cache_key,
                llm_response_content
            )

            return llm_response_content

        except Exception as e:
            self.logger.error(
                f"LLM调用失败:{str(e)}"
            )
            return ""

    def _item_name_filte_expr(self, validate_item_names: List[str]) -> str:
        quoted = ", ".join(f'"{v}"' for v in validate_item_names)
        return f" item_name in [{quoted}]"


if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging
    import json

    setup_logging()

    print("=" * 60)
    print("开始测试: HyDE 检索节点 (HydeSearchNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "RS-12 数字万用表如何测量直流电压？",
        "item_names": ["RS-12 数字万用表"],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  商品: {mock_state['item_names']}")
    print("-" * 60)

    node = HyDeSearchNode()
    result = node.process(mock_state)

    chunks = result.get("hyde_embedding_chunks", [])
    print(f"\n【HyDE 检索结果】: {len(chunks)} 条")
    for i, chunk in enumerate(chunks, 1):
        entity = chunk.get("entity", {})
        print(f"  [{i}] chunk_id={entity.get('chunk_id')} "
              f"item_name={entity.get('item_name')} "
              f"distance={chunk.get('distance', 'N/A')}")
        content = entity.get("content", "")
        print(f"      内容: {content[:80]}...")

    hyde_doc = result.get("hyde_doc", "")
    if hyde_doc:
        print(f"\n【假设性文档】:\n{hyde_doc[:200]}...")

    print("-" * 60)
    print("测试完成")
