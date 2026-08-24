"""
RAG Retrieval 离线评测脚本

评测目标：
    在已经知道正确商品 item_name 的前提下，
    评估知识库 Retrieval Pipeline 能否召回正确 chunk。

当前评测链路：

    evaluation_dataset
            ↓
       question
       item_name
            ↓
       multi_search
            ↓
    ┌───────┼────────┐
    ↓       ↓        ↓
 Vector   HyDE       KG
    └───────┼────────┘
            ↓
           RRF
            ↓
         Rerank
            ↓
           END


注意：

1. 不经过 ItemNameConfirmNode
   因为商品识别和 Retrieval 是两个不同的评测任务。

2. 不经过 MCP Web Search
   当前评测的是本地知识库 Retrieval 能力。

3. 不经过 AnswerOutputNode
   当前只评估检索，不评估最终答案生成。

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
import hashlib
import argparse
from datetime import datetime
from typing import Dict, Any, List

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from knowledge.processor.query_process.state import QueryGraphState

from knowledge.processor.query_process.nodes.vector_search_node import (
    VectorSearchNode
)

from knowledge.processor.query_process.nodes.dense_search_node import (
    DenseSearchNode
)

from knowledge.processor.query_process.nodes.hyde_search_node import (
    HyDeSearchNode
)

from knowledge.processor.query_process.nodes.kg_search_node import (
    KnowledgeGraphSearchNode
)

from knowledge.processor.query_process.nodes.rrf_node import (
    RrfNode
)

from knowledge.processor.query_process.nodes.rerank_node import (
    RerankNode
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
from knowledge.processor.query_process.config import get_config

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
    "final_ablation"
)


def configure_evaluation_paths(dataset: str, result_subdir: str) -> None:
    """显式选择评测集，并保证结果仍写在 evaluation/results 下。"""
    global DATASET_PATH, RESULT_DIR

    dataset_path = (
        dataset
        if os.path.isabs(dataset)
        else os.path.join(CURRENT_DIR, "datasets", dataset)
    )
    dataset_path = os.path.abspath(dataset_path)
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"找不到评测集：{dataset_path}")

    results_root = os.path.abspath(os.path.join(CURRENT_DIR, "results"))
    result_dir = os.path.abspath(os.path.join(results_root, result_subdir))
    if os.path.commonpath([results_root, result_dir]) != results_root:
        raise ValueError("result_subdir 必须位于 evaluation/results 下")

    DATASET_PATH = dataset_path
    RESULT_DIR = result_dir


# ============================================================
# 评测参数
# ============================================================

K_VALUES = [1, 3, 5]

MRR_K = 5

RRF_WEIGHTS = {
    "hybrid": 1.0,
    "hyde": 0.9,
    "kg": 0.4,
}

MODE_SPECS = {
    "dense": {
        "group": "A",
        "label": "Dense",
        "output_field": "reranked_docs",
        "active_retrievers": ["dense"],
        "use_rrf": False,
        "use_rerank": False,
        "use_cliff_cutoff": False,
    },
    "hybrid": {
        "group": "B",
        "label": "Hybrid",
        "output_field": "embedding_chunks",
        "active_retrievers": ["hybrid"],
        "use_rrf": False,
        "use_rerank": False,
        "use_cliff_cutoff": False,
    },
    "hybrid_hyde": {
        "group": "C",
        "label": "Hybrid + HyDE + RRF",
        "output_field": "rrf_chunks",
        "active_retrievers": ["hybrid", "hyde"],
        "use_rrf": True,
        "use_rerank": False,
        "use_cliff_cutoff": False,
    },
    "hybrid_hyde_kg": {
        "group": "D",
        "label": "Hybrid + HyDE + KG + RRF",
        "output_field": "rrf_chunks",
        "active_retrievers": ["hybrid", "hyde", "kg"],
        "use_rrf": True,
        "use_rerank": False,
        "use_cliff_cutoff": False,
    },
    "full": {
        "group": "E",
        "label": "Full (RRF + CrossEncoder + cliff cutoff)",
        "output_field": "reranked_docs",
        "active_retrievers": ["hybrid", "hyde", "kg"],
        "use_rrf": True,
        "use_rerank": True,
        "use_cliff_cutoff": True,
    },
    "rrf_rerank_no_cutoff": {
        "group": "F",
        "label": "RRF + CrossEncoder (ranking only)",
        "output_field": "reranked_docs",
        "active_retrievers": ["hybrid", "hyde", "kg"],
        "use_rrf": True,
        "use_rerank": True,
        "use_cliff_cutoff": False,
    },
    "rrf_rerank_cliff": {
        "group": "G",
        "label": "RRF + CrossEncoder + cliff cutoff",
        "output_field": "reranked_docs",
        "active_retrievers": ["hybrid", "hyde", "kg"],
        "use_rrf": True,
        "use_rerank": True,
        "use_cliff_cutoff": True,
    },
}

FINAL_MODES = list(MODE_SPECS)


# ============================================================
# 1. 创建 Retrieval Evaluation Graph
# ============================================================
class NormalizeRetrievalNode(BaseNode):

    name = "normalize_retrieval_node"

    def __init__(self, source_field: str):
        super().__init__()
        self.source_field = source_field

    def process(self, state):

        if self.source_field not in state:
            raise RuntimeError(
                f"评测数据流错误: state 中缺少 {self.source_field}"
            )

        chunks = state.get(self.source_field) or []

        state["reranked_docs"] = chunks

        return state


class EvaluationRrfNode(RrfNode):
    """使用显式、可落盘的权重执行与生产节点相同的 RRF 公式。"""

    def process(self, state: QueryGraphState) -> QueryGraphState:
        rrf_inputs = [
            (
                self._normalize_input(state.get("embedding_chunks") or []),
                RRF_WEIGHTS["hybrid"],
            ),
            (
                self._normalize_input(state.get("hyde_embedding_chunks") or []),
                RRF_WEIGHTS["hyde"],
            ),
            (
                self._normalize_input(state.get("kg_chunks") or []),
                RRF_WEIGHTS["kg"],
            ),
        ]
        merged = self._rrf_merge(rrf_inputs, self._rrf_k, self._top_k)
        state["rrf_chunks"] = [doc for doc, _ in merged]
        return state


class EvaluationRerankNode(RerankNode):
    """只为消融实验提供 cliff cutoff 开关，不改变生产节点。"""

    def __init__(self, apply_cliff_cutoff: bool):
        super().__init__()
        self.apply_cliff_cutoff = apply_cliff_cutoff

    def process(self, state: QueryGraphState) -> QueryGraphState:
        user_query = state.get("rewritten_query", "") or state.get("original_query", "")
        merged_docs = self._merge_multi_source_docs(state)
        ranked_docs = self._rerank_merged_docs(user_query, merged_docs)
        state["reranked_docs"] = (
            self._cliff_cutoff(ranked_docs)
            if self.apply_cliff_cutoff
            else ranked_docs[:self.config.rerank_max_top_k]
        )
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
def create_retrieval_eval_graph(mode="full") -> CompiledStateGraph:
    """
    创建 Retrieval Ablation Evaluation Graph

    五组实验：

    dense:
        Dense Search

    hybrid:
        Dense + Sparse Hybrid

    hybrid_hyde:
        Hybrid + HyDE + RRF

    hybrid_hyde_kg:
        Hybrid + HyDE + KG + RRF

    full:
        Hybrid + HyDE + KG + RRF + Rerank
    """

    # ========================================================
    # A: Dense Only
    # ========================================================
    if mode == "dense":

        workflow = StateGraph(QueryGraphState)

        workflow.add_node(
            "dense_search",
            DenseSearchNode()
        )

        workflow.set_entry_point(
            "dense_search"
        )

        workflow.add_edge(
            "dense_search",
            END
        )

        return workflow.compile()


    # ========================================================
    # B: Hybrid(Dense + Sparse)
    # ========================================================
    elif mode == "hybrid":

        workflow = StateGraph(QueryGraphState)

        workflow.add_node(
            "search_embedding",
            VectorSearchNode()
        )

        workflow.add_node(
            "normalize_result",
            NormalizeRetrievalNode("embedding_chunks")
        )

        workflow.set_entry_point(
            "search_embedding"
        )

        workflow.add_edge(
            "search_embedding",
            "normalize_result"
        )

        workflow.add_edge(
            "normalize_result",
            END
        )

        return workflow.compile()


    # ========================================================
    # C: Hybrid + HyDE + RRF
    # ========================================================
    elif mode == "hybrid_hyde":

        workflow = StateGraph(QueryGraphState)


        nodes = {

            "multi_search": lambda x: x,

            "search_embedding": VectorSearchNode(),

            "search_embedding_hyde": HyDeSearchNode(),

            "join": lambda x: {},

            "rrf": EvaluationRrfNode(),

            "normalize_result": NormalizeRetrievalNode("rrf_chunks")
        }


        for name, node in nodes.items():

            workflow.add_node(
                name,
                node
            )


        workflow.set_entry_point(
            "multi_search"
        )


        # 并行召回
        workflow.add_edge(
            "multi_search",
            "search_embedding"
        )

        workflow.add_edge(
            "multi_search",
            "search_embedding_hyde"
        )


        # 汇合
        workflow.add_edge(
            "search_embedding",
            "join"
        )

        workflow.add_edge(
            "search_embedding_hyde",
            "join"
        )


        workflow.add_edge(
            "join",
            "rrf"
        )


        workflow.add_edge(
            "rrf",
            "normalize_result"
        )


        workflow.add_edge(
            "normalize_result",
            END
        )


        return workflow.compile()



    # ========================================================
    # D: Hybrid + HyDE + KG + RRF
    # ========================================================
    elif mode == "hybrid_hyde_kg":

        workflow = StateGraph(QueryGraphState)


        nodes = {

            "multi_search": lambda x: x,

            "search_embedding": VectorSearchNode(),

            "search_embedding_hyde": HyDeSearchNode(),

            "query_kg": KnowledgeGraphSearchNode(),

            "join": lambda x: {},

            "rrf": EvaluationRrfNode(),

            "normalize_result": NormalizeRetrievalNode("rrf_chunks")
        }


        for name, node in nodes.items():

            workflow.add_node(
                name,
                node
            )


        workflow.set_entry_point(
            "multi_search"
        )


        # 三路召回
        workflow.add_edge(
            "multi_search",
            "search_embedding"
        )

        workflow.add_edge(
            "multi_search",
            "search_embedding_hyde"
        )

        workflow.add_edge(
            "multi_search",
            "query_kg"
        )


        # 汇合
        workflow.add_edge(
            "search_embedding",
            "join"
        )

        workflow.add_edge(
            "search_embedding_hyde",
            "join"
        )

        workflow.add_edge(
            "query_kg",
            "join"
        )


        workflow.add_edge(
            "join",
            "rrf"
        )


        workflow.add_edge(
            "rrf",
            "normalize_result"
        )


        workflow.add_edge(
            "normalize_result",
            END
        )


        return workflow.compile()



    # ========================================================
    # E: Full System
    # Hybrid + HyDE + KG + RRF + Rerank
    # ========================================================
    elif mode in {
        "full",
        "rrf_rerank_no_cutoff",
        "rrf_rerank_cliff",
    }:

        workflow = StateGraph(QueryGraphState)


        nodes = {

            "multi_search": lambda x: x,

            "search_embedding": VectorSearchNode(),

            "search_embedding_hyde": HyDeSearchNode(),

            "query_kg": KnowledgeGraphSearchNode(),

            "join": lambda x: {},

            "rrf": EvaluationRrfNode(),

            "rerank": EvaluationRerankNode(
                apply_cliff_cutoff=(mode != "rrf_rerank_no_cutoff")
            )
        }


        for name, node in nodes.items():

            workflow.add_node(
                name,
                node
            )


        workflow.set_entry_point(
            "multi_search"
        )


        # 三路召回
        workflow.add_edge(
            "multi_search",
            "search_embedding"
        )

        workflow.add_edge(
            "multi_search",
            "search_embedding_hyde"
        )

        workflow.add_edge(
            "multi_search",
            "query_kg"
        )


        # 汇合
        workflow.add_edge(
            "search_embedding",
            "join"
        )

        workflow.add_edge(
            "search_embedding_hyde",
            "join"
        )

        workflow.add_edge(
            "query_kg",
            "join"
        )


        workflow.add_edge(
            "join",
            "rrf"
        )


        workflow.add_edge(
            "rrf",
            "rerank"
        )


        workflow.add_edge(
            "rerank",
            END
        )


        return workflow.compile()


    else:

        raise ValueError(
            f"Unsupported evaluation mode: {mode}"
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
# 3. 提取 Rerank 后的 chunk_id
# ============================================================

def extract_chunk_ids(
        reranked_docs: List[Dict[str, Any]]
) -> List[str]:
    """
    从 reranked_docs 中按照当前排序顺序提取 chunk_id。

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
        sample: Dict[str, Any],
        mode: str
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

    # 保存各阶段的 chunk_id，并从当前模式声明的字段读取最终结果。
    stage_chunk_ids = {
        field: extract_chunk_ids(result.get(field) or [])
        for field in (
            "embedding_chunks",
            "hyde_embedding_chunks",
            "kg_chunks",
            "rrf_chunks",
            "reranked_docs",
        )
        if field in result
    }

    output_field = MODE_SPECS[mode]["output_field"]
    if output_field not in result:
        raise RuntimeError(
            f"评测数据流错误: {mode} 未产生声明的输出字段 {output_field}"
        )

    output_docs = result.get(output_field) or []

    # --------------------------------------------------------
    # 提取最终 chunk_id
    # --------------------------------------------------------

    retrieved_chunk_ids = extract_chunk_ids(
        output_docs
    )

    # --------------------------------------------------------
    # 打印检索结果
    # --------------------------------------------------------

    print(
        f"\nRerank 返回数量: "
        f"{len(retrieved_chunk_ids)}"
    )

    print(
        "Rerank Chunk IDs:"
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

        "output_field": output_field,

        "stage_chunk_ids": stage_chunk_ids,

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

def run_evaluation(
        mode="full",
        dataset=None,
        warmup=True,
        suite_timestamp=None
):
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

    if mode not in MODE_SPECS:
        raise ValueError(f"Unsupported evaluation mode: {mode}")

    if warmup:
        warmup_embedding_model()
    # ========================================================
    # 1. 加载数据集
    # ========================================================

    if dataset is None:
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

            sample=sample,

            mode=mode
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

        "group": MODE_SPECS[mode]["group"],

        "label": MODE_SPECS[mode]["label"],

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

    timestamp = suite_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    result_path = os.path.join(

        RESULT_DIR,

        (
            f"retrieval_eval_"
            f"{MODE_SPECS[mode]['group']}_"
            f"{mode}_"
            f"{timestamp}.json"
        )
    )

    config = get_config()
    parameters = {
        **MODE_SPECS[mode],
        "rrf": {
            "k": config.rrf_k,
            "max_results": config.rrf_max_results,
            "weights": RRF_WEIGHTS,
            "active_weights": {
                name: RRF_WEIGHTS[name]
                for name in MODE_SPECS[mode]["active_retrievers"]
                if name in RRF_WEIGHTS
            },
        } if MODE_SPECS[mode]["use_rrf"] else None,
        "rerank": {
            "max_top_k": config.rerank_max_top_k,
            "min_top_k": config.rerank_min_top_k,
            "gap_abs": config.rerank_gap_abs,
            "gap_ratio": config.rerank_gap_ratio,
            "apply_cliff_cutoff": MODE_SPECS[mode]["use_cliff_cutoff"],
        } if MODE_SPECS[mode]["use_rerank"] else None,
    }

    output = {

        "mode": mode,

        "group": MODE_SPECS[mode]["group"],

        "label": MODE_SPECS[mode]["label"],

        "dataset_path": os.path.abspath(DATASET_PATH),

        "parameters": parameters,

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

    output["result_path"] = os.path.abspath(result_path)
    return output


def _mode_parameters(mode: str) -> Dict[str, Any]:
    config = get_config()
    spec = MODE_SPECS[mode]
    return {
        **spec,
        "rrf": {
            "k": config.rrf_k,
            "max_results": config.rrf_max_results,
            "weights": dict(RRF_WEIGHTS),
            "active_weights": {
                name: RRF_WEIGHTS[name]
                for name in spec["active_retrievers"]
                if name in RRF_WEIGHTS
            },
        } if spec["use_rrf"] else None,
        "rerank": {
            "max_top_k": config.rerank_max_top_k,
            "min_top_k": config.rerank_min_top_k,
            "gap_abs": config.rerank_gap_abs,
            "gap_ratio": config.rerank_gap_ratio,
            "apply_cliff_cutoff": spec["use_cliff_cutoff"],
        } if spec["use_rerank"] else None,
    }


def _score_question(
        sample: Dict[str, Any],
        retrieved_chunk_ids: List[str],
        output_field: str,
        stage_chunk_ids: Dict[str, List[str]]
) -> Dict[str, Any]:
    gold_chunk_ids = [str(chunk_id) for chunk_id in sample["gold_chunk_ids"]]
    return {
        "id": sample.get("id", ""),
        "question": sample.get("question", ""),
        "category": sample.get("category", ""),
        "item_name": sample.get("item_name", ""),
        "gold_titles": sample.get("gold_titles", []),
        "gold_chunk_ids": gold_chunk_ids,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "output_field": output_field,
        "stage_chunk_ids": stage_chunk_ids,
        "recall_at_1": recall_at_k(retrieved_chunk_ids, gold_chunk_ids, 1),
        "recall_at_3": recall_at_k(retrieved_chunk_ids, gold_chunk_ids, 3),
        "recall_at_5": recall_at_k(retrieved_chunk_ids, gold_chunk_ids, 5),
        "rr_at_5": reciprocal_rank(retrieved_chunk_ids, gold_chunk_ids, MRR_K),
    }


def _summarize_questions(mode: str, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "mode": mode,
        "group": MODE_SPECS[mode]["group"],
        "label": MODE_SPECS[mode]["label"],
        "dataset_size": len(questions),
        "mean_recall_at_1": sum(q["recall_at_1"] for q in questions) / len(questions),
        "mean_recall_at_3": sum(q["recall_at_3"] for q in questions) / len(questions),
        "mean_recall_at_5": sum(q["recall_at_5"] for q in questions) / len(questions),
        "mrr_at_5": sum(q["rr_at_5"] for q in questions) / len(questions),
    }


def _sequence_digest(questions: List[Dict[str, Any]]) -> str:
    payload = [
        [question["id"], question["retrieved_chunk_ids"]]
        for question in questions
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compare_mode_chunk_ids(
        results_by_mode: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    comparisons = []
    for left_index, left_mode in enumerate(FINAL_MODES):
        left_questions = results_by_mode[left_mode]
        for right_mode in FINAL_MODES[left_index + 1:]:
            right_questions = results_by_mode[right_mode]
            different_count = sum(
                left["retrieved_chunk_ids"] != right["retrieved_chunk_ids"]
                for left, right in zip(left_questions, right_questions)
            )
            comparisons.append({
                "left_group": MODE_SPECS[left_mode]["group"],
                "left_mode": left_mode,
                "right_group": MODE_SPECS[right_mode]["group"],
                "right_mode": right_mode,
                "different_question_count": different_count,
                "identical_all_questions": different_count == 0,
            })

    f_questions = results_by_mode["rrf_rerank_no_cutoff"]
    g_questions = results_by_mode["rrf_rerank_cliff"]
    cliff_is_prefix = all(
        f["retrieved_chunk_ids"][:len(g["retrieved_chunk_ids"])]
        == g["retrieved_chunk_ids"]
        for f, g in zip(f_questions, g_questions)
    )

    return {
        "sequence_sha256": {
            mode: _sequence_digest(questions)
            for mode, questions in results_by_mode.items()
        },
        "pairwise": comparisons,
        "data_flow_assertions": {
            "B_reads_embedding_chunks": all(
                q["retrieved_chunk_ids"] == q["stage_chunk_ids"]["embedding_chunks"]
                for q in results_by_mode["hybrid"]
            ),
            "C_reads_rrf_chunks": all(
                q["retrieved_chunk_ids"] == q["stage_chunk_ids"]["rrf_chunks"]
                for q in results_by_mode["hybrid_hyde"]
            ),
            "D_reads_rrf_chunks": all(
                q["retrieved_chunk_ids"] == q["stage_chunk_ids"]["rrf_chunks"]
                for q in results_by_mode["hybrid_hyde_kg"]
            ),
            "E_F_G_read_reranked_docs": all(
                q["retrieved_chunk_ids"] == q["stage_chunk_ids"]["reranked_docs"]
                for mode in ("full", "rrf_rerank_no_cutoff", "rrf_rerank_cliff")
                for q in results_by_mode[mode]
            ),
            "G_is_prefix_of_F": cliff_is_prefix,
            "E_equals_G_expected_alias": (
                _sequence_digest(results_by_mode["full"])
                == _sequence_digest(results_by_mode["rrf_rerank_cliff"])
            ),
        },
    }


def _candidate_pool_diagnostics(
        questions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """区分“候选池带来新 gold”与“融合排序把 gold 排进 Top-K”。"""

    def union_ids(question, fields):
        return list(dict.fromkeys(
            chunk_id
            for field in fields
            for chunk_id in question["stage_chunk_ids"][field]
        ))

    def pool_recall(question, fields):
        gold_ids = set(question["gold_chunk_ids"])
        return len(set(union_ids(question, fields)) & gold_ids) / len(gold_ids)

    hybrid_fields = ["embedding_chunks"]
    hyde_fields = ["embedding_chunks", "hyde_embedding_chunks"]
    all_fields = ["embedding_chunks", "hyde_embedding_chunks", "kg_chunks"]
    return {
        "metric_scope": "unordered recall over the complete candidate union; not Recall@K",
        "mean_hybrid_candidate_recall": sum(
            pool_recall(question, hybrid_fields) for question in questions
        ) / len(questions),
        "mean_hybrid_hyde_union_recall": sum(
            pool_recall(question, hyde_fields) for question in questions
        ) / len(questions),
        "mean_hybrid_hyde_kg_union_recall": sum(
            pool_recall(question, all_fields) for question in questions
        ) / len(questions),
        "questions_where_hyde_adds_new_gold": sum(
            bool(
                (set(question["stage_chunk_ids"]["hyde_embedding_chunks"])
                 & set(question["gold_chunk_ids"]))
                - set(question["stage_chunk_ids"]["embedding_chunks"])
            )
            for question in questions
        ),
        "questions_where_kg_adds_new_gold_beyond_hybrid_hyde": sum(
            bool(
                (set(question["stage_chunk_ids"]["kg_chunks"])
                 & set(question["gold_chunk_ids"]))
                - set(union_ids(question, hyde_fields))
            )
            for question in questions
        ),
    }


def run_final_ablation_suite() -> Dict[str, Any]:
    """一次召回、一次 CrossEncoder 打分，派生 A-G 的受控最终消融。"""
    print("=" * 80)
    print("Shopkeeper Brain Final Controlled Ablation Evaluation")
    print("=" * 80)

    warmup_embedding_model()
    dataset = load_dataset()
    if not dataset:
        raise RuntimeError("评测集为空")

    dense_graph = create_retrieval_eval_graph(mode="dense")
    multi_retrieval_graph = create_retrieval_eval_graph(mode="hybrid_hyde_kg")
    rrf_node = EvaluationRrfNode()
    rerank_node = EvaluationRerankNode(apply_cliff_cutoff=False)
    results_by_mode = {mode: [] for mode in FINAL_MODES}

    for index, sample in enumerate(dataset, start=1):
        question_id = sample.get("id", "")
        question = sample.get("question", "")
        item_name = sample.get("item_name", "")
        gold_chunk_ids = sample.get("gold_chunk_ids") or []
        if not question or not item_name or not gold_chunk_ids:
            raise ValueError(f"{question_id} 缺少 question、item_name 或 gold_chunk_ids")

        print(f"[{index}/{len(dataset)}] {question_id}: {question}")
        base_state = {
            "original_query": question,
            "rewritten_query": question,
            "item_names": [item_name],
            "session_id": f"final_ablation_{question_id}_{uuid.uuid4().hex}",
            "task_id": f"final_ablation_task_{uuid.uuid4().hex}",
            "is_stream": False,
        }

        # A 单独执行 Dense；B/C/D 共用一次 Hybrid/HyDE/KG 召回。
        dense_state = dense_graph.invoke(dict(base_state))
        shared_state = multi_retrieval_graph.invoke(dict(base_state))

        dense_ids = extract_chunk_ids(dense_state.get("reranked_docs") or [])
        hybrid_ids = extract_chunk_ids(shared_state.get("embedding_chunks") or [])
        hyde_ids = extract_chunk_ids(shared_state.get("hyde_embedding_chunks") or [])
        kg_ids = extract_chunk_ids(shared_state.get("kg_chunks") or [])
        rrf_d_ids = extract_chunk_ids(shared_state.get("rrf_chunks") or [])

        # C 只在 D 的共享召回结果上移除 KG 输入，其他输入与参数完全不变。
        c_state = dict(shared_state)
        c_state["kg_chunks"] = []
        c_state = rrf_node.process(c_state)
        rrf_c_ids = extract_chunk_ids(c_state.get("rrf_chunks") or [])

        # F/G 共用一次 CrossEncoder 打分；G 只多做 cliff cutoff。
        merged_docs = rerank_node._merge_multi_source_docs(shared_state)
        ranked_docs = rerank_node._rerank_merged_docs(question, merged_docs)
        f_docs = ranked_docs[:rerank_node.config.rerank_max_top_k]
        g_docs = rerank_node._cliff_cutoff(ranked_docs)
        f_ids = extract_chunk_ids(f_docs)
        g_ids = extract_chunk_ids(g_docs)

        common_retrieval_stages = {
            "embedding_chunks": hybrid_ids,
            "hyde_embedding_chunks": hyde_ids,
            "kg_chunks": kg_ids,
        }
        mode_ids_and_stages = {
            "dense": (dense_ids, {"reranked_docs": dense_ids}),
            "hybrid": (
                hybrid_ids,
                {**common_retrieval_stages, "reranked_docs": hybrid_ids},
            ),
            "hybrid_hyde": (
                rrf_c_ids,
                {**common_retrieval_stages, "rrf_chunks": rrf_c_ids, "reranked_docs": rrf_c_ids},
            ),
            "hybrid_hyde_kg": (
                rrf_d_ids,
                {**common_retrieval_stages, "rrf_chunks": rrf_d_ids, "reranked_docs": rrf_d_ids},
            ),
            "full": (
                g_ids,
                {**common_retrieval_stages, "rrf_chunks": rrf_d_ids, "reranked_docs": g_ids},
            ),
            "rrf_rerank_no_cutoff": (
                f_ids,
                {**common_retrieval_stages, "rrf_chunks": rrf_d_ids, "reranked_docs": f_ids},
            ),
            "rrf_rerank_cliff": (
                g_ids,
                {**common_retrieval_stages, "rrf_chunks": rrf_d_ids, "reranked_docs": g_ids},
            ),
        }

        for mode, (retrieved_ids, stages) in mode_ids_and_stages.items():
            results_by_mode[mode].append(_score_question(
                sample=sample,
                retrieved_chunk_ids=retrieved_ids,
                output_field=MODE_SPECS[mode]["output_field"],
                stage_chunk_ids=stages,
            ))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULT_DIR, exist_ok=True)
    dataset_path = os.path.abspath(DATASET_PATH)
    with open(DATASET_PATH, "rb") as dataset_file:
        dataset_sha256 = hashlib.sha256(dataset_file.read()).hexdigest()

    mode_outputs = {}
    for mode in FINAL_MODES:
        group = MODE_SPECS[mode]["group"]
        mode_output = {
            "mode": mode,
            "group": group,
            "label": MODE_SPECS[mode]["label"],
            "dataset_path": dataset_path,
            "dataset_sha256": dataset_sha256,
            "execution_design": "shared retrieval once; shared CrossEncoder scores for F/G; E is G alias",
            "parameters": _mode_parameters(mode),
            "summary": _summarize_questions(mode, results_by_mode[mode]),
            "questions": results_by_mode[mode],
        }
        result_path = os.path.join(
            RESULT_DIR,
            f"final_ablation_{group}_{mode}_{timestamp}.json",
        )
        mode_output["result_path"] = os.path.abspath(result_path)
        with open(result_path, "w", encoding="utf-8") as result_file:
            json.dump(mode_output, result_file, ensure_ascii=False, indent=2)
        mode_outputs[mode] = mode_output

    chunk_id_checks = _compare_mode_chunk_ids(results_by_mode)
    combined_output = {
        "experiment": "shopkeeper_brain_final_controlled_ablation",
        "timestamp": timestamp,
        "dataset_path": dataset_path,
        "dataset_sha256": dataset_sha256,
        "dataset_size": len(dataset),
        "execution_design": {
            "retrieval_runs_per_question": 1,
            "dense_runs_per_question": 1,
            "cross_encoder_score_runs_per_question": 1,
            "E_is_alias_of_G": True,
        },
        "summaries": [mode_outputs[mode]["summary"] for mode in FINAL_MODES],
        "candidate_pool_diagnostics": _candidate_pool_diagnostics(
            results_by_mode["hybrid_hyde_kg"]
        ),
        "mode_result_files": {
            mode: mode_outputs[mode]["result_path"]
            for mode in FINAL_MODES
        },
        "chunk_id_checks": chunk_id_checks,
    }
    combined_path = os.path.join(RESULT_DIR, f"final_ablation_summary_{timestamp}.json")
    combined_output["result_path"] = os.path.abspath(combined_path)
    with open(combined_path, "w", encoding="utf-8") as combined_file:
        json.dump(combined_output, combined_file, ensure_ascii=False, indent=2)

    print("\n最终消融汇总")
    for summary in combined_output["summaries"]:
        print(
            f"{summary['group']} {summary['label']}: "
            f"R@1={summary['mean_recall_at_1']:.4f}, "
            f"R@3={summary['mean_recall_at_3']:.4f}, "
            f"R@5={summary['mean_recall_at_5']:.4f}, "
            f"MRR@5={summary['mrr_at_5']:.4f}"
        )
    print(f"汇总结果: {combined_path}")
    return combined_output


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="运行 A-G 受控检索消融实验")
    parser.add_argument(
        "--dataset",
        default="KG专项evaluation_dataset_40_final.json",
        help="evaluation/datasets 下的数据集文件名，或绝对路径",
    )
    parser.add_argument(
        "--result-subdir",
        default="final_ablation",
        help="evaluation/results 下的独立结果子目录",
    )
    args = parser.parse_args()
    configure_evaluation_paths(args.dataset, args.result_subdir)

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

    run_final_ablation_suite()


