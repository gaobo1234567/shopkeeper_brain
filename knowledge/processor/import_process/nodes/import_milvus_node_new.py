import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from typing import Sequence, List, Any, Dict, Optional
from dataclasses import dataclass
from pymilvus import DataType, MilvusClient
from pymilvus.orm.schema import CollectionSchema
from knowledge.processor.import_process.base import BaseNode, setup_logging, T
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError, EmbeddingError
from knowledge.processor.import_process.config import get_config
from knowledge.utils.milvus_util import get_milvus_client


@dataclass(frozen=True)
class ScalarFieldSpec:
    field_name: str
    datatype: DataType
    max_length: Optional[int] = None


_SCALAR_FIELDS: Sequence[ScalarFieldSpec] = (
    ScalarFieldSpec(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535),
)


class _MilvusSchemaBuilder:
    @staticmethod
    def build(client: MilvusClient, dim: int) -> CollectionSchema:
        logger.info("开始构建约束(schema)...")

        schema = client.create_schema(enable_dynamic_field=True)

        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True
        )

        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=dim
        )

        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )

        for scalar_field in _SCALAR_FIELDS:
            kwargs: Dict[str, Any] = {
                "field_name": scalar_field.field_name,
                "datatype": scalar_field.datatype
            }
            if scalar_field.max_length is not None:
                kwargs["max_length"] = scalar_field.max_length
            schema.add_field(**kwargs)

        logger.info("构建约束(schema)完成...")
        return schema


class _MilvusIndexBuilder:
    @staticmethod
    def build(client: MilvusClient, collection_name: str):
        logger.info(f"开始构建集合 {collection_name} 索引...")

        index = client.prepare_index_params(collection_name=collection_name)

        index.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )

        index.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        logger.info(f"构建集合 {collection_name} 索引完成...")
        return index


class _MilvusInserter:
    """
    职责：
    1. 将数据分批插入到Milvus
    2. 回填chunk_id
    3. 返回实际成功插入的记录数
    """

    def __init__(self, client: MilvusClient, collection_name: str, batch_size: int = 500):
        self._client = client
        self._collection_name = collection_name
        self._batch_size = batch_size

    def insert(self, chunks: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
        logger.info(
            f"开始分批插入 {len(chunks)} 块到Milvus，batch_size={self._batch_size}..."
        )

        total_inserted_count = 0

        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start:start + self._batch_size]

            inserted_result = self._client.insert(
                collection_name=self._collection_name,
                data=batch
            )

            inserted_count = inserted_result.get("insert_count", 0)
            ids = inserted_result.get("ids", [])

            self._fill_chunk_ids(batch, ids)
            total_inserted_count += inserted_count

            logger.info(
                f"Milvus批量插入完成：{start + 1}-{start + len(batch)}/{len(chunks)}，"
                f"本批成功插入={inserted_count}"
            )

        logger.info(
            f"Milvus全部批次处理完成：待插入={len(chunks)}，成功插入={total_inserted_count}"
        )

        return chunks, total_inserted_count

    def _fill_chunk_ids(self, chunks: List[Dict[str, Any]], ids: List[Any]):
        for chunk, milvus_id in zip(chunks, ids):
            chunk["chunk_id"] = milvus_id


class ImportMilvusNode(BaseNode):
    name = "import_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        validated_chunks, dim, config, total_chunk_count = self._validate_get_inputs(state)

        milvus_client = get_milvus_client()
        if milvus_client is None:
            return state

        collection = getattr(config, "chunks_collection")

        self._ensure_has_collection(
            milvus_client=milvus_client,
            collection_name=collection,
            dim=dim
        )

        milvus_insert_batch_size = getattr(config, "milvus_insert_batch_size", 500)
        if not isinstance(milvus_insert_batch_size, int) or milvus_insert_batch_size <= 0:
            self.logger.warning(
                f"milvus_insert_batch_size={milvus_insert_batch_size} 无效，自动使用默认值500"
            )
            milvus_insert_batch_size = 500

        inserter = _MilvusInserter(
            client=milvus_client,
            collection_name=collection,
            batch_size=milvus_insert_batch_size
        )

        final_chunks, inserted_count = inserter.insert(chunks=validated_chunks)

        valid_count = len(validated_chunks)
        invalid_count = total_chunk_count - valid_count

        valid_ratio = valid_count / total_chunk_count if total_chunk_count else 0
        insert_ratio_total = inserted_count / total_chunk_count if total_chunk_count else 0
        insert_ratio_valid = inserted_count / valid_count if valid_count else 0

        self.logger.info(
            "Milvus导入统计："
            f"原始Chunk总数={total_chunk_count}, "
            f"向量有效={valid_count}/{total_chunk_count} ({valid_ratio:.2%}), "
            f"向量无效跳过={invalid_count}, "
            f"成功插入={inserted_count}/{total_chunk_count} ({insert_ratio_total:.2%}), "
            f"有效Chunk插入成功率={inserted_count}/{valid_count} ({insert_ratio_valid:.2%})"
        )

        state["chunks"] = final_chunks
        return state

    def _validate_get_inputs(self, state: ImportGraphState):
        self.log_step("step1", "参数校验")

        config = get_config()
        chunks = state.get("chunks")

        if not chunks:
            raise ValidationError("待入库的切块chunk不存在", self.name)

        total_chunk_count = len(chunks)
        validated_chunks = []

        for chunk in chunks:
            if chunk.get("dense_vector") and chunk.get("sparse_vector"):
                validated_chunks.append(chunk)
            else:
                self.logger.error("待入库的切块chunk的混合向量不存在，该chunk跳过入库")

        if not validated_chunks:
            raise ValidationError("入库的chunk都无效", self.name)

        dim = len(validated_chunks[0].get("dense_vector"))

        self.logger.info(
            f"导入Milvus向量数据库的有效块：{len(validated_chunks)}/{total_chunk_count}，"
            f"chunk的稠密向量维度={dim}"
        )

        return validated_chunks, dim, config, total_chunk_count

    def _ensure_has_collection(
        self,
        milvus_client: MilvusClient,
        collection_name: str,
        dim: int,
        delete_flag: bool = False
    ):
        self.log_step("step2", f"准备集合 {collection_name} 创建")

        # 默认不删除已有collection，避免导入新文档时清空旧知识库。
        if delete_flag and milvus_client.has_collection(collection_name=collection_name):
            self.logger.warning(f"显式要求删除Milvus集合：{collection_name}")
            milvus_client.drop_collection(collection_name=collection_name)

        if milvus_client.has_collection(collection_name=collection_name):
            self.logger.info(f"{collection_name} 集合已经存在，直接复用")
            return

        schema = _MilvusSchemaBuilder.build(milvus_client, dim)
        index = _MilvusIndexBuilder.build(milvus_client, collection_name)

        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index
        )

        self.logger.info(f"{collection_name} 集合创建完成")


from pathlib import Path
import json


def _cli_main() -> None:
    setup_logging()

    temp_dir = Path(
        r"D:\pycharmprojects\shopkeeper_brain\knowledge\processor\import_process\output_temp_dir\万用表的使用\hybrid_auto"
    )

    input_path = temp_dir / "chunks_vector.json"
    output_path = temp_dir / "chunks_vector_ids.json"

    if not input_path.exists():
        logger.error(f"找不到输入文件: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as fh:
        content = json.load(fh)

    state: ImportGraphState = {
        "chunks": content.get("chunks", [])
    }

    import_milvus = ImportMilvusNode()
    result_state = import_milvus.process(state)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result_state, fh, ensure_ascii=False, indent=4)

    logger.info(f"备份临时文件{output_path}成功")


if __name__ == "__main__":
    _cli_main()