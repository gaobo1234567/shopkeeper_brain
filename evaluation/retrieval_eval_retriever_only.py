"""
RAG Retrieval 离线评测脚本

评测目标：
    在已经知道正确商品 item_name 的前提下，
    评估知识库 Retrieval Pipeline 能否召回正确 chunk。

当前评测链路：

    1. hybrid: Hybrid(Dense + Sparse) Only
    2. hyde:   HyDE Only
    3. kg:     Knowledge Graph Only

    三组实验彼此独立，均不经过 RRF 和 Rerank。

评测指标：

    Recall@1
    Recall@3
    Recall@5
    MRR@5


执行：

    python evaluation/retrieval_eval.py
"""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from knowledge.processor.query_process.state import QueryGraphState

from knowledge.processor.query_process.nodes.vector_search_node import (
    VectorSearchNode
)


from knowledge.processor.query_process.nodes.hyde_search_node import (
    HyDeSearchNode
)

from knowledge.processor.query_process.nodes.kg_search_node import (
    KnowledgeGraphSearchNode
)



from evaluation.metrics import (
    recall_at_k,
    reciprocal_rank,
    mean_recall_at_k,
    mean_reciprocal_rank
)
from knowledge.utils.bge_m3_embedding_util import (
    get_beg_m3_embedding_model,
    generate_hybrid_embeddings
)
from knowledge.processor.query_process.base import BaseNode

# ============================================================
# 路径配置
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    CURRENT_DIR,
    "datasets",
    "KG专项evaluation_dataset_40_final.json"
)

RESULT_DIR = os.path.join(
    CURRENT_DIR,
    "results",
    "RRF_weight"
)


# ============================================================
# 评测参数
# ============================================================

K_VALUES = [1, 3, 5]

MRR_K = 5


# ============================================================
# 1. 创建 Retrieval Evaluation Graph
# ============================================================
class NormalizeRetrievalNode(BaseNode):

    name = "normalize_retrieval_node"

    def __init__(self, source_key: str):
        super().__init__()
        self.source_key = source_key

    def process(self, state):
        """
        将当前独立 Retriever 的原始结果统一放到 reranked_docs。

        注意：这里仅复用原评测代码的统一结果字段，
        当前三组实验都没有真正执行 Rerank。
        """
        chunks = state.get(self.source_key, []) or []
        state["reranked_docs"] = chunks
        return state


def warmup_embedding_model():
    """
    单线程预热 BGE-M3。

    原正式 Query Graph 中，ItemNameConfirmNode 会在多路并行检索前
    提前使用 embedding model，因此模型已经完成初始化。

    Retrieval Evaluation 绕过 ItemNameConfirm 后，
    Vector / HyDE / KG 可能在并行阶段同时首次访问 BGE-M3，
    在 Windows + PyTorch / FlagEmbedding 环境下可能触发
    native access violation。

    因此在正式开始评测前，主动单线程执行一次真实 embedding。
    """

    print("=" * 80)
    print("正在预热 BGE-M3 embedding model...")
    print("=" * 80)

    embedding_model = get_beg_m3_embedding_model()

    if embedding_model is None:
        raise RuntimeError(
            "BGE-M3 embedding model 初始化失败"
        )

    # 不只是调用 getter。
    # 必须真正执行一次 embedding，
    # 才能让 sparse_linear / colbert_linear 等组件完成初始化。
    result = generate_hybrid_embeddings(
        embedding_model,
        embedding_documents=[
            "BGE-M3 retrieval evaluation warmup"
        ]
    )

    if not result:
        raise RuntimeError(
            "BGE-M3 warmup embedding 执行失败"
        )

    print("BGE-M3 warmup 完成")
def create_retrieval_eval_graph(mode="hybrid") -> CompiledStateGraph:
    """
    创建三组独立 Retrieval Evaluation Graph。

    hybrid: 只执行 Dense + Sparse Hybrid Search
    hyde:   只执行 HyDE Search
    kg:     只执行 Knowledge Graph Search

    三组实验都不经过 RRF，也不经过 Rerank。
    """

    if mode == "hybrid":
        workflow = StateGraph(QueryGraphState)
        workflow.add_node("search_embedding", VectorSearchNode())
        workflow.add_node(
            "normalize_result",
            NormalizeRetrievalNode(source_key="embedding_chunks")
        )
        workflow.set_entry_point("search_embedding")
        workflow.add_edge("search_embedding", "normalize_result")
        workflow.add_edge("normalize_result", END)
        return workflow.compile()

    elif mode == "hyde":
        workflow = StateGraph(QueryGraphState)
        workflow.add_node("search_embedding_hyde", HyDeSearchNode())
        workflow.add_node(
            "normalize_result",
            NormalizeRetrievalNode(source_key="hyde_embedding_chunks")
        )
        workflow.set_entry_point("search_embedding_hyde")
        workflow.add_edge("search_embedding_hyde", "normalize_result")
        workflow.add_edge("normalize_result", END)
        return workflow.compile()

    elif mode == "kg":
        workflow = StateGraph(QueryGraphState)
        workflow.add_node("query_kg", KnowledgeGraphSearchNode())
        workflow.add_node(
            "normalize_result",
            NormalizeRetrievalNode(source_key="kg_chunks")
        )
        workflow.set_entry_point("query_kg")
        workflow.add_edge("query_kg", "normalize_result")
        workflow.add_edge("normalize_result", END)
        return workflow.compile()

    else:
        raise ValueError(
            f"Unsupported evaluation mode: {mode}. "
            f"Supported modes: hybrid / hyde / kg"
        )


# ============================================================
# 2. 读取评测数据集
# ============================================================

def load_dataset() -> List[Dict[str, Any]]:
    """
    读取：

        evaluation/datasets/evaluation_dataset.json

    返回：

        [
            {
                "id": "q001",
                "question": "...",
                "item_name": "...",
                "gold_chunk_ids": [...]
            },
            ...
        ]
    """

    if not os.path.exists(
            DATASET_PATH
    ):

        raise FileNotFoundError(
            f"找不到评测集：{DATASET_PATH}"
        )

    with open(
            DATASET_PATH,
            "r",
            encoding="utf-8"
    ) as f:

        dataset = json.load(f)

    if not isinstance(
            dataset,
            list
    ):

        raise ValueError(
            "evaluation_dataset.json "
            "最外层必须是 JSON List"
        )

    return dataset


# ============================================================
# 3. 提取 Retriever 返回的 chunk_id
# ============================================================

def extract_chunk_ids(
        reranked_docs: List[Dict[str, Any]]
) -> List[str]:
    """
    从当前 Retriever 返回结果中按照当前排序顺序提取 chunk_id。

    所有 chunk_id 都统一转换为字符串。

    原因：

        JSON：
            468489822519646585

        数据库返回：
            "468489822519646585"

    Python 中：

        123 != "123"

    如果不统一类型，
    会导致正确召回却被判断成未命中。
    """

    chunk_ids = []

    for doc in reranked_docs:

        if not isinstance(
                doc,
                dict
        ):
            continue

        # ----------------------------------------------------
        # 正常情况
        # ----------------------------------------------------

        chunk_id = doc.get(
            "chunk_id"
        )

        # ----------------------------------------------------
        # 兼容某些可能的嵌套结构
        #
        # {
        #     "entity": {
        #         "chunk_id": ...
        #     }
        # }
        # ----------------------------------------------------

        if chunk_id is None:

            entity = doc.get(
                "entity"
            )

            if isinstance(
                    entity,
                    dict
            ):

                chunk_id = entity.get(
                    "chunk_id"
                )

        # ----------------------------------------------------
        # 加入结果
        # ----------------------------------------------------

        if chunk_id is not None:

            chunk_ids.append(
                str(chunk_id)
            )

    return chunk_ids


# ============================================================
# 4. 单道题评测
# ============================================================

def evaluate_one_question(
        graph: CompiledStateGraph,
        sample: Dict[str, Any]
) -> Dict[str, Any]:
    """
    对 evaluation_dataset 中的一道题执行 Retrieval Evaluation。
    """

    # --------------------------------------------------------
    # 基础字段
    # --------------------------------------------------------

    question_id = sample.get(
        "id",
        ""
    )

    question = sample.get(
        "question",
        ""
    )

    item_name = sample.get(
        "item_name",
        ""
    )

    category = sample.get(
        "category",
        ""
    )

    gold_titles = sample.get(
        "gold_titles",
        []
    )

    # --------------------------------------------------------
    # Gold chunk_id 统一为 str
    # --------------------------------------------------------

    gold_chunk_ids = [

        str(chunk_id)

        for chunk_id in sample.get(
            "gold_chunk_ids",
            []
        )
    ]

    # --------------------------------------------------------
    # 参数检查
    # --------------------------------------------------------

    if not question:

        raise ValueError(
            f"{question_id} 缺少 question"
        )

    if not item_name:

        raise ValueError(
            f"{question_id} 缺少 item_name"
        )

    if not gold_chunk_ids:

        raise ValueError(
            f"{question_id} 缺少 gold_chunk_ids"
        )

    # --------------------------------------------------------
    # 每道题使用独立 session
    #
    # 虽然当前已经绕过 ItemNameConfirm，
    # 但仍然保持评测样本之间完全隔离。
    # --------------------------------------------------------

    session_id = (
        f"retrieval_eval_"
        f"{question_id}_"
        f"{uuid.uuid4().hex}"
    )

    task_id = (
        f"retrieval_eval_task_"
        f"{uuid.uuid4().hex}"
    )

    # --------------------------------------------------------
    # 构造 QueryGraphState
    #
    # 关键：
    #
    # 1. original_query = 原始问题
    #
    # 2. rewritten_query = 原始问题
    #
    #    当前 benchmark 暂时不评 Query Rewrite，
    #    所以直接使用原问题。
    #
    # 3. item_names = Gold ItemName
    #
    #    直接使用人工标注的正确商品名，
    #    绕过 ItemNameConfirm。
    # --------------------------------------------------------

    state = {

        "original_query": question,

        "rewritten_query": question,

        "item_names": [
            item_name
        ],

        "session_id": session_id,

        "task_id": task_id,

        "is_stream": False
    }

    # --------------------------------------------------------
    # 打印当前题目
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"[{question_id}] "
        f"{question}"
    )

    print(
        f"Item Name: "
        f"{item_name}"
    )

    print(
        f"Gold Chunk IDs: "
        f"{gold_chunk_ids}"
    )

    # --------------------------------------------------------
    # 执行 Retrieval Graph
    # --------------------------------------------------------

    result = graph.invoke(
        state
    )

    # --------------------------------------------------------
    # 获取当前 Retriever 的原始结果
    # --------------------------------------------------------

    reranked_docs = result.get(
        "reranked_docs",
        []
    )

    if reranked_docs is None:

        reranked_docs = []

    # --------------------------------------------------------
    # 提取最终 chunk_id
    # --------------------------------------------------------

    retrieved_chunk_ids = extract_chunk_ids(
        reranked_docs
    )

    # --------------------------------------------------------
    # 打印检索结果
    # --------------------------------------------------------

    print(
        f"\nRetriever 返回数量: "
        f"{len(retrieved_chunk_ids)}"
    )

    print(
        "Retrieved Chunk IDs:"
    )

    if not retrieved_chunk_ids:

        print(
            "  无结果"
        )

    for rank, chunk_id in enumerate(
            retrieved_chunk_ids,
            start=1
    ):

        hit_mark = (

            " <-- GOLD"

            if chunk_id
            in gold_chunk_ids

            else ""
        )

        print(
            f"  Rank {rank}: "
            f"{chunk_id}"
            f"{hit_mark}"
        )

    # ========================================================
    # Recall@K
    # ========================================================

    recall_scores = {}

    for k in K_VALUES:

        recall_scores[k] = recall_at_k(

            retrieved_chunk_ids=(
                retrieved_chunk_ids
            ),

            gold_chunk_ids=(
                gold_chunk_ids
            ),

            k=k
        )

    # ========================================================
    # Reciprocal Rank
    # ========================================================

    rr_at_5 = reciprocal_rank(

        retrieved_chunk_ids=(
            retrieved_chunk_ids
        ),

        gold_chunk_ids=(
            gold_chunk_ids
        ),

        k=MRR_K
    )

    # --------------------------------------------------------
    # 打印当前题目的指标
    # --------------------------------------------------------

    print(
        "\n指标："
    )

    print(
        f"  Recall@1 = "
        f"{recall_scores[1]:.4f}"
    )

    print(
        f"  Recall@3 = "
        f"{recall_scores[3]:.4f}"
    )

    print(
        f"  Recall@5 = "
        f"{recall_scores[5]:.4f}"
    )

    print(
        f"  RR@5      = "
        f"{rr_at_5:.4f}"
    )

    # --------------------------------------------------------
    # 单题结果
    # --------------------------------------------------------

    return {

        "id": question_id,

        "question": question,

        "category": category,

        "item_name": item_name,

        "gold_titles": gold_titles,

        "gold_chunk_ids": (
            gold_chunk_ids
        ),

        "retrieved_chunk_ids": (
            retrieved_chunk_ids
        ),

        "recall_at_1": (
            recall_scores[1]
        ),

        "recall_at_3": (
            recall_scores[3]
        ),

        "recall_at_5": (
            recall_scores[5]
        ),

        "rr_at_5": rr_at_5
    }


# ============================================================
# 5. 整个数据集评测
# ============================================================

def run_evaluation(mode="hybrid"):
    """
    执行完整 Retrieval Evaluation。
    """

    print(
        "=" * 80
    )

    print(
        "Shopkeeper Brain "
        "Offline Retrieval Evaluation"
    )

    print(
        "=" * 80
    )
    # ========================================================
    # 0. 单线程预热 BGE-M3
    # ========================================================

    warmup_embedding_model()
    # ========================================================
    # 1. 加载数据集
    # ========================================================

    dataset = load_dataset()

    print(
        f"评测集问题数量: "
        f"{len(dataset)}"
    )

    # ========================================================
    # 2. 创建 Retrieval Graph
    #
    # Graph 只创建一次。
    # ========================================================

    graph = (
        create_retrieval_eval_graph(mode=mode)
    )

    # ========================================================
    # 3. 每道题执行评测
    # ========================================================

    question_results = []

    for index, sample in enumerate(
            dataset,
            start=1
    ):

        print(
            f"\n当前进度："
            f"{index}/{len(dataset)}"
        )

        result = evaluate_one_question(

            graph=graph,

            sample=sample
        )

        question_results.append(
            result
        )

    # ========================================================
    # 4. 准备 aggregate metrics
    # ========================================================

    all_retrieved = [

        result[
            "retrieved_chunk_ids"
        ]

        for result
        in question_results
    ]

    all_gold = [

        result[
            "gold_chunk_ids"
        ]

        for result
        in question_results
    ]

    # ========================================================
    # 5. Mean Recall@1
    # ========================================================

    mean_recall_1 = (
        mean_recall_at_k(

            all_retrieved_chunk_ids=(
                all_retrieved
            ),

            all_gold_chunk_ids=(
                all_gold
            ),

            k=1
        )
    )

    # ========================================================
    # 6. Mean Recall@3
    # ========================================================

    mean_recall_3 = (
        mean_recall_at_k(

            all_retrieved_chunk_ids=(
                all_retrieved
            ),

            all_gold_chunk_ids=(
                all_gold
            ),

            k=3
        )
    )

    # ========================================================
    # 7. Mean Recall@5
    # ========================================================

    mean_recall_5 = (
        mean_recall_at_k(

            all_retrieved_chunk_ids=(
                all_retrieved
            ),

            all_gold_chunk_ids=(
                all_gold
            ),

            k=5
        )
    )

    # ========================================================
    # 8. MRR@5
    # ========================================================

    mrr_5 = (
        mean_reciprocal_rank(

            all_retrieved_chunk_ids=(
                all_retrieved
            ),

            all_gold_chunk_ids=(
                all_gold
            ),

            k=5
        )
    )

    # ========================================================
    # 9. Summary
    # ========================================================

    summary = {

        "mode": mode,

        "dataset_size": (
            len(dataset)
        ),

        "mean_recall_at_1": (
            mean_recall_1
        ),

        "mean_recall_at_3": (
            mean_recall_3
        ),

        "mean_recall_at_5": (
            mean_recall_5
        ),

        "mrr_at_5": (
            mrr_5
        )
    }

    # ========================================================
    # 10. 打印总结果
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "最终 Retrieval Evaluation 结果"
    )

    print(
        "=" * 80
    )

    print(
        f"Dataset Size   : "
        f"{len(dataset)}"
    )

    print(
        f"Mean Recall@1  : "
        f"{mean_recall_1:.4f}"
    )

    print(
        f"Mean Recall@3  : "
        f"{mean_recall_3:.4f}"
    )

    print(
        f"Mean Recall@5  : "
        f"{mean_recall_5:.4f}"
    )

    print(
        f"MRR@5          : "
        f"{mrr_5:.4f}"
    )

    # ========================================================
    # 11. 保存结果
    # ========================================================

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    timestamp = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    result_path = os.path.join(

        RESULT_DIR,

        (
            f"retrieval_eval_"
            f"{mode}_"
            f"{timestamp}.json"
        )
    )

    output = {

        "summary": summary,

        "questions": (
            question_results
        )
    }

    with open(
            result_path,
            "w",
            encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # 12. 输出保存路径
    # ========================================================

    print(
        "\n评测结果已保存到："
    )

    print(
        result_path
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 如果你希望看到各节点完整日志，
    # 可以保留项目本身的 logging 配置。
    # --------------------------------------------------------

    try:

        from knowledge.processor.query_process.base import setup_logging

        setup_logging()

    except Exception:

        # logging 初始化失败不影响 evaluation 主流程
        pass

    # 三组独立实验：每次只运行其中一种模式即可
    # run_evaluation(mode="hybrid")
    # run_evaluation(mode="hyde")
    run_evaluation(mode="kg")

