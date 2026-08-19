import os, json
from typing import Dict, List, Any
from pathlib import Path
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError, EmbeddingError
from knowledge.processor.import_process.config import get_config
from knowledge.utils.bge_m3_embedding_util import get_beg_m3_embedding_model


class BgeEmbeddingChunksNode(BaseNode):

    """
    BgeEmbeddingChunksNode主要职责：
    1. 获取所有的chunks拼接要向量的内容
    2. 批量嵌入 chunk的（embedding_content:item_name + chunk.get('content')）
    补充：加入item_name的目的
    （1）把唯一能代表的上下文信息尽量的都注入到块中或者其他位置，辅助检索质量更好。
        放在元数据中按照标量字段进行检索。import_milvus_node.py
        注入到向量内容中去。bge_embedding_chunks_node.py
    （2）检索的时候问题中也会提取出商品名字和导入的时候保存的商品名字进行比较。
    3. 将所有chunk嵌入后的向量值，存储到列表中，在返回给下一个节点用
    """

    name = "bge_embedding_chunks_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 参数校验
        validated_chunks, config = self._validate_get_inputs(state)

        # 2. 获取批量嵌入的阈值
        embedding_batch_chunk_size = getattr(config, 'embedding_batch_size', 16)

        # 3. 准备分批嵌入(pineline)
        total_length = len(validated_chunks)
        final_chunks = []
        for i in range(0, total_length, embedding_batch_chunk_size):
            batch = validated_chunks[i:i + embedding_batch_chunk_size]
            # 拼接要嵌入的内容 向量嵌入的内容 把嵌入的向量注入到chunk中
            batch_chunks = self._process_batch_chunks(batch, i, total_length)
            final_chunks.extend(batch_chunks)

        # 4. 统计向量生成情况
        total_chunks = len(final_chunks)
        dense_success_count = sum(
            1 for chunk in final_chunks
            if chunk.get('dense_vector')
        )
        sparse_success_count = sum(
            1 for chunk in final_chunks
            if chunk.get('sparse_vector')
        )
        both_success_count = sum(
            1 for chunk in final_chunks
            if chunk.get('dense_vector') and chunk.get('sparse_vector')
        )

        dense_success_ratio = dense_success_count / total_chunks if total_chunks else 0
        sparse_success_ratio = sparse_success_count / total_chunks if total_chunks else 0
        both_success_ratio = both_success_count / total_chunks if total_chunks else 0

        self.logger.info(
            f"Chunk向量生成统计：总数={total_chunks}, "
            f"Dense成功={dense_success_count}/{total_chunks} ({dense_success_ratio:.2%}), "
            f"Sparse成功={sparse_success_count}/{total_chunks} ({sparse_success_ratio:.2%}), "
            f"Dense+Sparse同时成功={both_success_count}/{total_chunks} ({both_success_ratio:.2%})"
        )

        # 5. 更新&返回state
        state['chunks'] = final_chunks
        return state

    def _validate_get_inputs(self, state: ImportGraphState):
        config = get_config()

        self.log_step("step1", "参数校验")
        chunks = state.get('chunks')

        if not chunks or not isinstance(chunks, list):
            raise ValidationError(f"chunks为空或者无效", self.name)

        self.logger.info(f"嵌入的块数：{len(chunks)}")
        return chunks, config

    def _process_batch_chunks(self, batch: List[Dict[str, Any]], star_index: int, total_length: int):

        self.log_step("step2", f"开始批量处理chunk嵌入:批次{star_index + 1}-{star_index + len(batch)}")
        embedding_contents = []
        for chunk in batch:
            content = chunk.get('content')
            item_name = chunk.get('item_name') or ""
            embedding_content = f"{item_name}\n{content}"
            embedding_contents.append(embedding_content)

        try:
            bge_m3_model = get_beg_m3_embedding_model()
            embedding_result = bge_m3_model.encode_documents(documents=embedding_contents)

            if not embedding_result:
                self.logger.warning(f"嵌入后的结果不存在...")
                return batch
        except Exception as e:
            self.logger.warning(f"嵌入向量嵌入失败...{str(e)}")
            return batch

        for index, chunk in enumerate(batch):
            dense_vector = embedding_result['dense'][index].tolist()

            csr_array = embedding_result['sparse']
            ind_ptr = csr_array.indptr
            start_ind_ptr = ind_ptr[index]
            end_ind_ptr = ind_ptr[index + 1]
            token_id = csr_array.indices[start_ind_ptr:end_ind_ptr].tolist()
            weight = csr_array.data[start_ind_ptr:end_ind_ptr].tolist()
            sparse_vector = dict(zip(token_id, weight))

            chunk['dense_vector'] = dense_vector
            chunk['sparse_vector'] = sparse_vector

        self.logger.info(f"开始批量处理chunk嵌入:批次{star_index + 1}-{star_index + len(batch)}/{total_length}")
        return batch


if __name__ == '__main__':
    setup_logging()

    base_temp_dir = Path(
        r"D:\pycharmprojects\shopkeeper_brain\knowledge\processor\import_process\output_temp_dir\万用表的使用\hybrid_auto")

    input_path = base_temp_dir / "chunks.json"
    output_path = base_temp_dir / "chunks_vector.json"

    if not input_path.exists():
        print(f" 找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    state = {
        "chunks": content
    }

    node_bge_embedding = BgeEmbeddingChunksNode()
    proceed_result = node_bge_embedding.process(state)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(proceed_result, f, ensure_ascii=False, indent=4)

    print(f" 向量生成测试完成！结果已成功备份至:\n{output_path}")