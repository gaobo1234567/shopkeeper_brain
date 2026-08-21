from typing import List, Sequence, Union


ChunkId = Union[int, str]


def recall_at_k(
        retrieved_chunk_ids: Sequence[ChunkId],
        gold_chunk_ids: Sequence[ChunkId],
        k: int
) -> float:
    """
    计算 Recall@K。

    含义：
        在前 K 个检索结果中，
        一共召回了多少比例的 gold chunk。

    公式：
        Recall@K =
        Top-K 中命中的 gold chunk 数量
        /
        gold chunk 总数量

    示例1：
        retrieved = [10, 20, 30, 40, 50]
        gold = [30]

        Recall@5 = 1 / 1 = 1.0

    示例2：
        retrieved = [10, 20, 30, 40, 50]
        gold = [30, 60]

        Recall@5 = 1 / 2 = 0.5

    Args:
        retrieved_chunk_ids:
            当前检索阶段按照排名从高到低返回的 chunk_id。

        gold_chunk_ids:
            人工标注的正确 chunk_id。

        k:
            只考虑前 K 个检索结果。

    Returns:
        float:
            Recall@K，范围 [0, 1]。
    """

    # ---------------------------------------------------------
    # 1. 参数保护
    # ---------------------------------------------------------

    if k <= 0:
        return 0.0

    if not gold_chunk_ids:
        return 0.0

    # ---------------------------------------------------------
    # 2. 只取 Top-K
    # ---------------------------------------------------------

    top_k_ids = retrieved_chunk_ids[:k]

    # ---------------------------------------------------------
    # 3. 转成 set，方便做集合交集
    #
    # 例如：
    #
    # retrieved:
    # [A, B, C, D, E]
    #
    # gold:
    # [C, F]
    #
    # intersection:
    # {C}
    # ---------------------------------------------------------

    top_k_set = set(top_k_ids)
    gold_set = set(gold_chunk_ids)

    hit_gold_ids = top_k_set & gold_set

    # ---------------------------------------------------------
    # 4. Recall@K
    # ---------------------------------------------------------

    recall = len(hit_gold_ids) / len(gold_set)

    return recall


def reciprocal_rank(
        retrieved_chunk_ids: Sequence[ChunkId],
        gold_chunk_ids: Sequence[ChunkId],
        k: int = None
) -> float:
    """
    计算单个问题的 Reciprocal Rank（RR）。

    含义：
        看第一个正确的 chunk 排在第几名。

    公式：
        RR = 1 / rank

    示例1：

        retrieved:
        [A, B, C, D]

        gold:
        [C]

        C 排名第 3

        RR = 1 / 3


    示例2：

        retrieved:
        [C, A, B]

        gold:
        [C]

        C 排名第 1

        RR = 1


    示例3：

        retrieved:
        [A, B, C, D]

        gold:
        [C, D]

        第一个相关结果 C 排名第 3

        RR = 1 / 3

    注意：
        MRR 只关心：
        “第一个正确结果出现在哪里”。

        不关心后面还有多少正确结果。

    Args:
        retrieved_chunk_ids:
            按相关性从高到低排列的检索结果。

        gold_chunk_ids:
            正确 chunk_id，可以有多个。

        k:
            如果指定，只在前 K 个结果中计算。
            如果不指定，则检查全部 retrieved 结果。

    Returns:
        float:
            Reciprocal Rank。
            如果没有任何 gold chunk 被检索到，返回 0。
    """

    if not gold_chunk_ids:
        return 0.0

    # ---------------------------------------------------------
    # 1. 根据 k 决定检查范围
    # ---------------------------------------------------------

    if k is not None:

        if k <= 0:
            return 0.0

        retrieved_chunk_ids = retrieved_chunk_ids[:k]

    # ---------------------------------------------------------
    # 2. gold 转成 set
    # ---------------------------------------------------------

    gold_set = set(gold_chunk_ids)

    # ---------------------------------------------------------
    # 3. 从 Rank 1 开始找第一个正确结果
    # ---------------------------------------------------------

    for rank, chunk_id in enumerate(
            retrieved_chunk_ids,
            start=1
    ):

        if chunk_id in gold_set:

            return 1.0 / rank

    # ---------------------------------------------------------
    # 4. 一个正确结果都没找到
    # ---------------------------------------------------------

    return 0.0


def mean_reciprocal_rank(
        all_retrieved_chunk_ids: Sequence[Sequence[ChunkId]],
        all_gold_chunk_ids: Sequence[Sequence[ChunkId]],
        k: int = None
) -> float:
    """
    计算整个评测集的 MRR。

    MRR = Mean Reciprocal Rank

    即：

        每道题先计算 RR
        ↓
        所有 RR 求平均


    示例：

        Question 1:
            第一个正确结果排名 1
            RR = 1

        Question 2:
            第一个正确结果排名 2
            RR = 1/2

        Question 3:
            没有找到
            RR = 0

        MRR =
            (1 + 0.5 + 0) / 3
            = 0.5


    Args:
        all_retrieved_chunk_ids:
            所有问题的检索结果。

            例如：

            [
                [1, 2, 3, 4],
                [10, 11, 12],
                [20, 21, 22]
            ]


        all_gold_chunk_ids:
            每一道题对应的 gold chunk。

            例如：

            [
                [2],
                [11],
                [99]
            ]


        k:
            如果指定，只在每道题的 Top-K 中计算 RR。

    Returns:
        float:
            整个评测集的 MRR。
    """

    # ---------------------------------------------------------
    # 1. 参数校验
    # ---------------------------------------------------------

    if not all_retrieved_chunk_ids:
        return 0.0

    if len(all_retrieved_chunk_ids) != len(all_gold_chunk_ids):

        raise ValueError(
            "all_retrieved_chunk_ids "
            "与 all_gold_chunk_ids 长度必须一致"
        )

    # ---------------------------------------------------------
    # 2. 每道题计算 RR
    # ---------------------------------------------------------

    rr_scores = []

    for retrieved_ids, gold_ids in zip(
            all_retrieved_chunk_ids,
            all_gold_chunk_ids
    ):

        rr = reciprocal_rank(
            retrieved_chunk_ids=retrieved_ids,
            gold_chunk_ids=gold_ids,
            k=k
        )

        rr_scores.append(rr)

    # ---------------------------------------------------------
    # 3. 求平均
    # ---------------------------------------------------------

    return sum(rr_scores) / len(rr_scores)


def mean_recall_at_k(
        all_retrieved_chunk_ids: Sequence[Sequence[ChunkId]],
        all_gold_chunk_ids: Sequence[Sequence[ChunkId]],
        k: int
) -> float:
    """
    计算整个评测集平均 Recall@K。

    每一道题：
        先计算 Recall@K

    然后：
        对所有问题取平均。


    Args:
        all_retrieved_chunk_ids:
            所有问题的检索结果。

        all_gold_chunk_ids:
            每道题对应的 gold chunk。

        k:
            Top-K。

    Returns:
        float:
            平均 Recall@K。
    """

    if not all_retrieved_chunk_ids:
        return 0.0

    if len(all_retrieved_chunk_ids) != len(all_gold_chunk_ids):

        raise ValueError(
            "all_retrieved_chunk_ids "
            "与 all_gold_chunk_ids 长度必须一致"
        )

    recall_scores = []

    for retrieved_ids, gold_ids in zip(
            all_retrieved_chunk_ids,
            all_gold_chunk_ids
    ):

        recall = recall_at_k(
            retrieved_chunk_ids=retrieved_ids,
            gold_chunk_ids=gold_ids,
            k=k
        )

        recall_scores.append(recall)

    return sum(recall_scores) / len(recall_scores)


# ============================================================
# 本地简单测试
# ============================================================

if __name__ == "__main__":

    # ---------------------------------------------------------
    # 模拟三道题
    # ---------------------------------------------------------

    retrieved_results = [
        # Q1：gold 在第1
        [101, 102, 103, 104, 105],

        # Q2：gold 在第3
        [201, 202, 203, 204, 205],

        # Q3：两个 gold，只召回一个
        [301, 302, 303, 304, 305]
    ]

    gold_results = [
        [101],
        [203],
        [303, 399]
    ]

    # ---------------------------------------------------------
    # Recall@5
    # ---------------------------------------------------------

    recall_5 = mean_recall_at_k(
        all_retrieved_chunk_ids=retrieved_results,
        all_gold_chunk_ids=gold_results,
        k=5
    )

    # ---------------------------------------------------------
    # MRR@5
    # ---------------------------------------------------------

    mrr_5 = mean_reciprocal_rank(
        all_retrieved_chunk_ids=retrieved_results,
        all_gold_chunk_ids=gold_results,
        k=5
    )

    print(f"Mean Recall@5: {recall_5:.4f}")
    print(f"MRR@5: {mrr_5:.4f}")