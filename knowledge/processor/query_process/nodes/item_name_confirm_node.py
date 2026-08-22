import logging
import json
import re

from json import JSONDecodeError
from typing import Dict, Any, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode

from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.milvus_util import (
    get_milvus_client,
    create_hybrid_search_requests,
    execute_hybrid_search_query
)
from knowledge.utils.bge_m3_embedding_util import (
    generate_hybrid_embeddings,
    get_beg_m3_embedding_model
)

from knowledge.prompts.query.query_prompt import ITEM_NAME_EXTRACT_TEMPLATE

from knowledge.utils.mongo_history_util import (
    get_recent_messages,
    update_message_item_names,

    # 新增：多轮澄清状态
    save_pending_clarification,
    get_pending_clarification,
    clear_pending_clarification
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 1. Item Name 对齐器
# ============================================================

class ItemNameAligner:
    """
    主要职责：

    1. 根据 LLM 提取的商品名称查询 Milvus
    2. 根据分数做 item_name 对齐
    3. 多个高分候选时判断：
       - 是否只是同一个商品的不同写法
       - 是否是真正不同的商品
    4. 对 confirmed 做分数差过滤
    """

    def match_align_filter(
            self,
            item_names: List[str]
    ) -> Tuple[List[str], List[str]]:

        # 1. 查询向量数据库
        search_result = self._match_vector(item_names)

        # 2. 分数对齐
        confirmed, options = self._item_name_score_align(search_result)

        # 3. confirmed 中如果存在多个商品，再做分数差异过滤
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(
                confirmed,
                search_result
            )

        return confirmed, options

    def _match_vector(
            self,
            item_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        根据 LLM 提取出来的商品名查询 knowledge_item_names。

        返回结构：

        [
            {
                "extracted_name": "HAK180 烫金机",
                "matches": [
                    {
                        "item_name": "hak180 烫金机",
                        "score": 0.9
                    }
                ]
            }
        ]
        """

        search_results = []

        # 1. Milvus
        milvus_client = get_milvus_client()

        if milvus_client is None:
            return search_results

        # 2. BGE-M3
        embedding_model = get_beg_m3_embedding_model()

        if embedding_model is None:
            logger.error("获取嵌入模型失败")
            return search_results

        # 3. 对 LLM 提取出来的 item_name 生成向量
        hybrid_embedding_result = generate_hybrid_embeddings(
            embedding_model,
            item_names
        )

        # 4. 每个 item_name 分别检索
        for index, extract_item_name in enumerate(item_names):

            hybrid_search_requests = create_hybrid_search_requests(
                dense_vector=hybrid_embedding_result["dense"][index],
                sparse_vector=hybrid_embedding_result["sparse"][index]
            )

            hybrid_search_result = execute_hybrid_search_query(
                milvus_client,
                collection_name="knowledge_item_names",
                search_requests=hybrid_search_requests,
                ranker_weights=(0.5, 0.5),
                norm_score=True,
                output_fields=["item_name"]
            )

            item_name_search_result = {
                "extracted_name": extract_item_name,

                "matches": [
                    {
                        "item_name": h["entity"]["item_name"],
                        "score": h["distance"]
                    }

                    for h in (
                        hybrid_search_result[0]
                        if hybrid_search_result
                        else []
                    )
                ]
            }

            search_results.append(item_name_search_result)

        return search_results

    # ========================================================
    # Canonical key
    # ========================================================

    def _canonical_item_key(self, item_name: str) -> str:
        """
        判断多个 item_name 是否实际上是同一商品。

        注意：
        该函数只用于“比较”。

        不修改数据库中真正保存的 item_name。

        例如：

        hak180 烫金机
        HAK 180 烫金机

        都会得到：

        hak180烫金机
        """

        if not item_name:
            return ""

        name = item_name.strip().lower()

        # 去除空格、横杠、下划线等格式差异
        name = re.sub(r"[\s\-_]+", "", name)

        return name

    def _item_name_score_align(
            self,
            search_results: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """
        根据 Milvus 检索分数将 item_name 分为：

        confirmed：
            可以直接进入后续检索。

        options：
            存在真正歧义，需要用户进一步选择。

        当前阈值：

        confirmed >= 0.7
        options   >= 0.6
        """

        confirmed = []
        options = []

        for item_name_search_result in search_results:

            extracted_name = item_name_search_result.get(
                "extracted_name"
            )

            matches = sorted(
                item_name_search_result.get("matches", []),
                key=lambda x: x["score"],
                reverse=True
            )

            # =================================================
            # 1. 高分候选
            # =================================================

            high = [
                m
                for m in matches
                if m.get("score", 0) >= 0.7
            ]

            if high:

                # ---------------------------------------------
                # A. LLM 提取名称和数据库名称完全一致
                # ---------------------------------------------

                extract = next(
                    (
                        h
                        for h in high
                        if str(h["item_name"]) == extracted_name
                    ),
                    None
                )

                if extract:

                    picked = extract["item_name"]

                    if picked not in confirmed:
                        confirmed.append(picked)

                # ---------------------------------------------
                # B. 只有一个高分候选
                # ---------------------------------------------

                elif len(high) == 1:

                    picked = high[0]["item_name"]

                    if picked not in confirmed:
                        confirmed.append(picked)

                # ---------------------------------------------
                # C. 多个高分候选
                # ---------------------------------------------

                else:

                    # 先判断：
                    #
                    # 多个候选究竟是：
                    #
                    # 同一商品的不同字符串写法
                    #
                    # 还是
                    #
                    # 真正不同的商品

                    canonical_groups = {}

                    for h in high[:3]:

                        picked = h.get("item_name")

                        if not picked:
                            continue

                        key = self._canonical_item_key(picked)

                        if key not in canonical_groups:
                            canonical_groups[key] = []

                        canonical_groups[key].append(picked)

                    logger.info(
                        f"high candidates: {high}"
                    )

                    logger.info(
                        f"canonical groups: {canonical_groups}"
                    )

                    # =========================================
                    # C1. 多个名字实际上属于同一个商品
                    # =========================================

                    if len(canonical_groups) == 1:

                        same_item_names = next(
                            iter(canonical_groups.values())
                        )

                        # 注意：
                        #
                        # 两个真实 item_name 都保留。
                        #
                        # 因为后续 Milvus filter 需要匹配真实数据库值。
                        #
                        # 例如：
                        #
                        # [
                        #   "hak180 烫金机",
                        #   "HAK 180 烫金机"
                        # ]

                        for picked in same_item_names:

                            if picked not in confirmed:
                                confirmed.append(picked)

                    # =========================================
                    # C2. 真正不同的商品
                    # =========================================

                    else:

                        for h in high[:3]:

                            picked = h.get("item_name")

                            if (
                                    picked
                                    and picked not in options
                                    and picked not in confirmed
                            ):
                                options.append(picked)

            # =================================================
            # 2. 没有高分候选，再判断中等分数
            # =================================================

            else:

                mid = [
                    m
                    for m in matches

                    if (
                            m.get("score", 0) >= 0.6
                            and m.get("item_name") not in options
                            and m.get("item_name") not in confirmed
                    )
                ]

                if mid:

                    for m in mid[:3]:

                        picked = m.get("item_name")

                        if picked:
                            options.append(picked)

        return confirmed, options[:3]

    def _item_name_score_filter(
            self,
            confirmed: List[str],
            search_results: List[Dict[str, Any]]
    ) -> List[str]:
        """
        对 confirmed 中的多个商品进一步过滤。

        规则：

        和最高分的差值 <= 0.15
        才保留。
        """

        item_name_score = {}

        for search_result in search_results:

            matches = search_result.get("matches", [])

            for m in matches:

                score = m.get("score", 0)
                item_name = m.get("item_name")

                if item_name in confirmed:

                    item_name_score[item_name] = max(
                        item_name_score.get(item_name) or 0,
                        score
                    )

        if not item_name_score:
            return confirmed

        sorted_item_name_score = sorted(
            item_name_score.items(),
            key=lambda x: x[1],
            reverse=True
        )

        max_item_name_score = sorted_item_name_score[0][1]

        return [
            name
            for name, score in item_name_score.items()

            if max_item_name_score - score <= 0.15
        ]


# ============================================================
# 2. 用户问题商品名称提取器
# ============================================================

class ItemNameExtractor:
    """
    根据：

    - 用户当前问题
    - 最近聊天历史

    提取：

    item_names
    rewritten_query
    """

    def extract_item_name(
            self,
            original_query: str,
            history_text: str
    ) -> Dict[str, Any]:

        result = {
            "item_names": [],
            "rewritten_query": original_query
        }

        llm_client = get_llm_client(response_format=True)

        if llm_client is None:
            return result

        human_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
            history_text=(
                history_text
                if history_text
                else "暂无上下文"
            ),
            query=original_query
        )

        system_prompt = (
            "你是一个专业的客服助手，"
            "擅长理解用户意图和提取关键信息。"
        )
        try:
            llm_response = llm_client.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt)
                ]
            )

            llm_content = getattr(
                llm_response,
                "content",
                ""
            ).strip()

            if not llm_content:
                return result

        except Exception as e:
            logger.warning(
                "商品名称提取 LLM 调用失败，使用默认结果降级: %s",
                e
            )
            return result
        try:
            parsed_result = self._clean_parse(llm_content)
            result["rewritten_query"] = (
                parsed_result.get("rewritten_query")
                or original_query
            )
            result["item_names"] = (
                parsed_result.get("item_names")
                or []
            )
        except Exception as e:

            logger.error(
                f"清洗以及解析LLM的输出失败: {str(e)}"
            )

        return result

    def _clean_parse(
            self,
            llm_response: str
    ) -> Dict[str, Any]:

        # 1. 去 JSON Markdown 围栏
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            llm_response.strip()
        )

        content = re.sub(
            r"\s*```$",
            "",
            cleaned
        )

        try:

            parsed_llm_result = json.loads(content)

            # ---------------------------------------------
            # item_names
            # ---------------------------------------------

            raw_item_names = parsed_llm_result.get(
                "item_names"
            )

            if not isinstance(raw_item_names, list):

                clean_item_names = []

            else:

                clean_item_names = [
                    raw_item.strip()

                    for raw_item in raw_item_names

                    if (
                            isinstance(raw_item, str)
                            and raw_item.strip()
                    )
                ]

            # ---------------------------------------------
            # rewritten_query
            # ---------------------------------------------

            raw_rewritten_query = parsed_llm_result.get(
                "rewritten_query"
            )

            clean_rewritten_query = (
                ""
                if not isinstance(raw_rewritten_query, str)
                else raw_rewritten_query.strip()
            )

            return {
                "item_names": clean_item_names,
                "rewritten_query": clean_rewritten_query
            }

        except JSONDecodeError as e:

            raise ValueError(
                f"JSON反序列LLM的输出失败：{str(e)}"
            )


# ============================================================
# 3. ItemNameConfirmNode
# ============================================================

class ItemNameConfirmNode(BaseNode):

    name = "item_name_confirm_node"

    def __init__(self):

        super().__init__()

        self._item_name_extractor = ItemNameExtractor()

        self._item_name_aligner = ItemNameAligner()

    # ========================================================
    # 解析用户对于 pending options 的选择
    # ========================================================

    def _parse_pending_selection(
            self,
            user_input: str,
            options: List[str]
    ) -> str:
        """
        将用户对上一轮 options 的回复解析成具体商品。

        支持：

        1
        2
        3

        第一个
        第二个
        第三个

        第1个
        第2个
        第3个

        选1
        选2

        商品完整名称

        商品唯一关键词，例如：
        G740
        """

        if not user_input or not options:
            return ""

        text = user_input.strip()

        # =====================================================
        # 1. 序号
        # =====================================================

        number_patterns = [
            r"^\s*(\d+)\s*$",
            r"^\s*第\s*(\d+)\s*个?\s*$",
            r"^\s*选\s*(\d+)\s*$",
            r"^\s*(\d+)\s*号\s*$"
        ]

        for pattern in number_patterns:

            match = re.match(pattern, text)

            if match:

                index = int(match.group(1)) - 1

                if 0 <= index < len(options):
                    return options[index]

        # =====================================================
        # 2. 中文序号
        # =====================================================

        chinese_index_map = {
            "第一个": 0,
            "第一": 0,
            "一": 0,

            "第二个": 1,
            "第二": 1,
            "二": 1,

            "第三个": 2,
            "第三": 2,
            "三": 2
        }

        normalized_text = re.sub(
            r"\s+",
            "",
            text
        )

        if normalized_text in chinese_index_map:

            index = chinese_index_map[normalized_text]

            if 0 <= index < len(options):
                return options[index]

        # =====================================================
        # 3. 商品名称直接匹配
        # =====================================================

        def normalize(value: str) -> str:

            value = value.strip().lower()

            return re.sub(
                r"[\s\-_]+",
                "",
                value
            )

        normalized_user_input = normalize(text)

        # 3.1 完全匹配
        for option in options:

            if normalize(option) == normalized_user_input:
                return option

        # =====================================================
        # 4. 唯一部分匹配
        #
        # 用户：
        # G740
        #
        # option：
        # 华为擎云 G740
        # =====================================================

        partial_matches = []

        for option in options:

            normalized_option = normalize(option)

            if (
                    normalized_user_input
                    and (
                        normalized_user_input in normalized_option
                        or normalized_option in normalized_user_input
                    )
            ):
                partial_matches.append(option)

        # 必须唯一，否则仍然有歧义
        if len(partial_matches) == 1:
            return partial_matches[0]

        return ""

    # ========================================================
    # 澄清提示
    # ========================================================

    def _build_option_prompt(
            self,
            options: List[str]
    ) -> str:
        """
        构建真正明确的商品选择提示。
        """

        option_lines = []

        for index, option in enumerate(
                options,
                start=1
        ):

            option_lines.append(
                f"{index}. {option}"
            )

        options_text = "\n".join(option_lines)

        return (
            "我找到了多个可能的产品，请选择您具体指的是哪一个：\n\n"
            f"{options_text}\n\n"
            "请回复产品名称或序号，例如“1”或“2”。"
        )

    # ========================================================
    # 主流程
    # ========================================================

    def process(
            self,
            state: QueryGraphState
    ) -> QueryGraphState:

        # -----------------------------------------------------
        # 1. 获取基础参数
        # -----------------------------------------------------

        original_query = state.get(
            "original_query",
            ""
        )

        session_id = state.get(
            "session_id",
            ""
        )

        # -----------------------------------------------------
        # 2. 获取最近聊天历史
        # -----------------------------------------------------

        chat_history = get_recent_messages(
            session_id,
            limit=10
        )

        history_text = ""

        for msg in chat_history:

            role = msg.get("role")

            content = msg.get(
                "text",
                ""
            )

            history_text += (
                f"{role}: {content}\n"
            )

        # =====================================================
        # 3. 第一优先级：
        #    检查上一轮是否正在等待用户选择商品
        # =====================================================

        pending_state = get_pending_clarification(
            session_id
        )

        if pending_state:

            pending_options = pending_state.get(
                "pending_options",
                []
            )

            pending_query = pending_state.get(
                "pending_query",
                ""
            )

            self.logger.info(
                f"检测到 pending clarification，"
                f"options={pending_options}"
            )

            # -------------------------------------------------
            # 3.1 用户是否取消本次选择
            # -------------------------------------------------

            cancel_words = {
                "取消",
                "算了",
                "不用了",
                "不问了"
            }

            if original_query.strip() in cancel_words:

                clear_pending_clarification(
                    session_id
                )

                state["answer"] = (
                    "已取消本次产品选择。"
                    "您可以重新提出问题。"
                )

                state["history"] = chat_history

                return state

            # -------------------------------------------------
            # 3.2 尝试解析用户选择
            # -------------------------------------------------

            selected_item = (
                self._parse_pending_selection(
                    original_query,
                    pending_options
                )
            )

            # -------------------------------------------------
            # 3.3 用户成功选中了某一个
            # -------------------------------------------------

            if selected_item:

                self.logger.info(
                    f"用户完成商品选择: {selected_item}"
                )

                # 清掉 pending 状态
                clear_pending_clarification(
                    session_id
                )

                # ---------------------------------------------
                # 非常关键：
                #
                # 用户第二轮可能只说：
                #
                # 2
                #
                # 真正要检索的是上一轮：
                #
                # 华为擎云怎么恢复系统？
                #
                # 所以恢复原始业务 query。
                # ---------------------------------------------

                effective_query = (
                    pending_query
                    if pending_query
                    else original_query
                )

                state["original_query"] = effective_query

                state["rewritten_query"] = effective_query

                state["item_names"] = [
                    selected_item
                ]

                # ---------------------------------------------
                # 给未绑定商品的历史消息回填 item_name
                # ---------------------------------------------

                ids_to_update = [
                    str(msg["_id"])

                    for msg in chat_history

                    if not msg.get("item_names")
                ]

                if ids_to_update:

                    try:

                        update_message_item_names(
                            ids_to_update,
                            [selected_item]
                        )

                    except Exception as e:

                        self.logger.warning(
                            f"回填历史 item_names 失败: {e}"
                        )

                state["history"] = chat_history

                return state

            # -------------------------------------------------
            # 3.4 用户输入无法唯一确定商品
            #
            # 例如：
            #
            # “是的”
            # -------------------------------------------------

            else:

                state["answer"] = (
                    "我还无法确定您具体选择的是哪一个产品。\n\n"
                    + self._build_option_prompt(
                        pending_options
                    )
                )

                state["history"] = chat_history

                return state

        # =====================================================
        # 4. 当前没有 pending clarification
        #    正常走原来的商品提取逻辑
        # =====================================================

        clean_llm_result = (
            self._item_name_extractor.extract_item_name(
                original_query,
                history_text
            )
        )

        item_names = clean_llm_result.get(
            "item_names",
            []
        )

        rewritten_query = clean_llm_result.get(
            "rewritten_query",
            original_query
        )

        # =====================================================
        # 5. Milvus item_name 对齐
        # =====================================================

        if item_names:

            confirmed, options = (
                self._item_name_aligner.match_align_filter(
                    item_names
                )
            )

        else:

            confirmed = []
            options = []

        # =====================================================
        # 6. 决策
        # =====================================================

        # -----------------------------------------------------
        # A. 已经确认商品
        # -----------------------------------------------------

        if confirmed:

            state["rewritten_query"] = rewritten_query

            state["item_names"] = confirmed

            # 回填历史消息 item_name
            ids_to_update = [
                str(msg["_id"])

                for msg in chat_history

                if not msg.get("item_names")
            ]

            if ids_to_update:

                try:

                    update_message_item_names(
                        ids_to_update,
                        confirmed
                    )

                except Exception as e:

                    self.logger.warning(
                        f"回填历史 item_names 失败: {e}"
                    )

        # -----------------------------------------------------
        # B. 真正存在多个歧义商品
        # -----------------------------------------------------

        elif options:

            # 保存跨轮澄清状态
            save_pending_clarification(
                session_id=session_id,
                pending_options=options,
                pending_query=(
                    rewritten_query
                    if rewritten_query
                    else original_query
                )
            )

            # 给用户返回真正明确的选择提示
            state["answer"] = (
                self._build_option_prompt(
                    options
                )
            )

        # -----------------------------------------------------
        # C. 完全识别不到商品
        # -----------------------------------------------------

        else:

            state["answer"] = (
                "抱歉，我无法识别您询问的具体产品名称，"
                "请提供更准确的产品名称或型号。"
            )

        # -----------------------------------------------------
        # 7. history 放入 state
        # -----------------------------------------------------

        state["history"] = chat_history

        return state


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":

    test_state: QueryGraphState = {
        "session_id": "test-session-001",

        "original_query": (
            "RS-12 数字万用表怎么测试电阻？"
        )
    }

    print(
        f"输入: "
        f"{json.dumps(test_state, ensure_ascii=False, indent=2)}"
    )

    node_item_name_confirm = (
        ItemNameConfirmNode()
    )

    result = (
        node_item_name_confirm.process(
            test_state
        )
    )

    print(
        f"确认商品: "
        f"{result.get('item_names')}"
    )

    print(
        f"改写查询: "
        f"{result.get('rewritten_query')}"
    )

    if result.get("answer"):

        print(
            f"拦截回复: "
            f"{result.get('answer')}"
        )