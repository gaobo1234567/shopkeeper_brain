import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple, Union
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.utils.bge_m3_embedding_util import get_beg_m3_embedding_model,generate_hybrid_embeddings
from knowledge.utils.milvus_util import get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query


class VectorSearchNode(BaseNode):
    name = "vector_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:

        try:
            # 1. 参数校验
            validated_query, validate_item_names = self._validate_query_inputs(state)
            # raise RuntimeError("模拟 Hybrid Milvus 服务故障")

            # 2. 获取嵌入模型 & Milvus 客户端
            embedding_model = get_beg_m3_embedding_model()
            milvus_client = get_milvus_client()

            if embedding_model is None or milvus_client is None:
                self.logger.warning(
                    "Hybrid Embedding 或 Milvus 客户端初始化失败，当前分支降级为空结果"
                )
                return {"embedding_chunks": []}

            # 3. 对问题向量化
            embedding_result = generate_hybrid_embeddings(
                embedding_model,
                embedding_documents=[validated_query]
            )

            if not embedding_result:
                self.logger.warning(
                    "Hybrid Embedding 生成失败，当前分支降级为空结果"
                )
                return {"embedding_chunks": []}

            # 4. 构建 item_name 过滤表达式
            item_name_filter_expr = self._item_name_filter(
                validate_item_names
            )

            # 5. 创建混合搜索请求
            hybrid_requests = create_hybrid_search_requests(
                dense_vector=embedding_result["dense"][0],
                sparse_vector=embedding_result["sparse"][0],
                expr=item_name_filter_expr,
                limit=5
            )

            # 6. 执行 Milvus 混合搜索
            reps = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=hybrid_requests,
                norm_score=True,
                output_fields=[
                    "chunk_id",
                    "content",
                    "item_name"
                ]
            )

            if not reps or not reps[0]:
                self.logger.warning(
                    "Hybrid 未检索到结果，当前分支返回空结果"
                )
                return {"embedding_chunks": []}

            # 7. 正常返回 Hybrid 检索结果
            return {
                "embedding_chunks": reps[0]
            }

        except StateFieldError:
            # State 字段错误属于程序逻辑问题，不做降级
            raise

        except Exception as e:
            # Embedding / Milvus 等运行时依赖异常：
            # 只让 Hybrid 当前分支失败，不影响 HyDE / KG
            self.logger.exception(
                "Hybrid Retriever 运行异常，当前分支降级为空结果: %s",
                e
            )

            return {
                "embedding_chunks": []
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

    # 使用 标量字段/动态字段 进行过滤检索
    def _item_name_filter(self, validate_item_names: List[str]) -> str:
        quoted = ", ".join(f'"{v}"' for v in validate_item_names)
        return f" item_name in [{quoted}]"


if __name__ == '__main__':
    state = {
        "rewritten_query": "万用表如何测量电阻",
        "item_names": ["RS-12 数字万用表"] #和数据库里的item_names对齐
    }

    vector_search = VectorSearchNode()

    result = vector_search.process(state)

    for r in result.get('embedding_chunks'):
        print(json.dumps(r, ensure_ascii=False, indent=2))
