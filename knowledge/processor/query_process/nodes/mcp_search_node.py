import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple, Union
from agents.mcp import MCPServerStreamableHttp

from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError


class McpSearchNode(BaseNode):
    name = "mcp_search_node"

    """
    负责从网络查询当前的问题【整个知识库没有找到该问题，兜底的网络结果】

    MCP 形式调用第三方的各种通用搜索工具。

    百度：
        商品比价工具
        商品搜索工具
        商品全维度对比工具
        商品下单工具
        百度搜索工具
        百度地图工具

    灵积服务平台：
        通用搜索工具【bailian_web_search】

    MCP 本质：
        各大平台把通用功能封装成工具，
        客户端通过 MCP 协议调用服务端。
    """

    def process(
        self,
        state: QueryGraphState
    ) -> Union[QueryGraphState, Dict[str, Any]]:

        try:
            # 1. 参数校验
            validated_rewritten_query, validated_item_names = (
                self._validate_query_inputs(state)
            )

            # 2. 获取 MCP Timeout
            # 如果 config 中暂时没有 mcp_timeout_seconds，
            # 默认使用 5 秒
            mcp_timeout_seconds = getattr(
                self.config,
                "mcp_timeout_seconds",
                8.0
            )

            # 3. 创建 MCP Client 并执行 bailian_web_search
            #
            # asyncio.wait_for：
            # 整个 MCP 搜索最多允许执行 mcp_timeout_seconds 秒
            mcp_result = asyncio.run(
                asyncio.wait_for(
                    self._create_execute_web_search(
                        validated_rewritten_query
                    ),
                    timeout=mcp_timeout_seconds
                )
            )

            # 4. MCP 没有返回有效结果
            if not mcp_result:
                return {
                    "web_search_docs": []
                }

            # 5. 只更新 state 的 web_search_docs
            return {
                "web_search_docs": mcp_result
            }

        except StateFieldError:
            # State 契约错误属于代码 / 数据流错误，
            # 不允许静默降级，继续向上抛出
            raise

        except asyncio.TimeoutError:
            # MCP 超时：
            # 当前检索分支降级为空，不影响其他检索分支
            self.logger.warning(
                "MCP Search 超时，当前检索分支降级为空结果"
            )

            return {
                "web_search_docs": []
            }

        except Exception as e:
            # MCP 连接失败、网络错误、服务端异常等运行时错误
            # 当前分支降级为空结果
            self.logger.exception(
                "MCP Search 运行异常，当前检索分支降级为空结果: %s",
                e
            )

            return {
                "web_search_docs": []
            }

    def _validate_query_inputs(
        self,
        state: QueryGraphState
    ) -> Tuple[str, List[str]]:

        # 1. 获取 state 的 rewritten_query
        rewritten_query = state.get("rewritten_query", "")

        # 2. 获取 state 的 item_names
        item_names = state.get("item_names", "")

        # 3. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(
                node_name=self.name,
                field_name="rewritten_query",
                expected_type=str
            )

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(
                node_name=self.name,
                field_name="item_names",
                expected_type=list
            )

        # 4. 返回
        return rewritten_query, item_names

    async def _create_execute_web_search(
        self,
        validated_rewritten_query: str
    ) -> List[Dict[str, Any]]:
        """
        1. 创建 MCP 客户端
        2. 建立 MCP 连接
        3. 调用 bailian_web_search
        4. 解析搜索结果
        5. 关闭连接
        """
        # await asyncio.sleep(10)
        # 1. 创建 MCP 客户端
        mcp_client = MCPServerStreamableHttp(
            name="通用搜索",
            params={
                "url": self.config.mcp_dashscope_base_url,
                "headers": {
                    "Authorization": self.config.mcp_dashscope_api_key
                }
            },
            cache_tools_list=True,
            # call_tool / list_tools 失败后最多重试 2 次
            max_retry_attempts=2,
            # 指数退避基础时间
            retry_backoff_seconds_base=0.5,
        )
        try:
            # 2. 建立 MCP 连接
            await mcp_client.connect()

            # 3. 执行工具
            execute_tool_result = await mcp_client.call_tool(
                tool_name="bailian_web_search",
                arguments={
                    "query": validated_rewritten_query,
                    "count": 2
                }
            )

            # 4. 解析执行结果

            # 4.1 没有返回结果
            if not execute_tool_result:
                return []

            # 4.2 content 不存在
            if (
                not execute_tool_result.content
                or not execute_tool_result.content[0]
            ):
                return []

            # 4.3 获取 TextContent.text
            text_content_text: str = (
                execute_tool_result.content[0].text
            )

            if not text_content_text:
                return []

            # 4.4 JSON 反序列化
            try:
                parsed_result: Dict[str, Any] = json.loads(
                    text_content_text
                )

                # 获取 pages
                pages = parsed_result.get("pages", [])

                if not pages:
                    return []

                search_result = []

                # 遍历每一个搜索结果
                for page in pages:
                    snippet = page.get("snippet", "").strip()
                    title = page.get("title", "").strip()
                    url = page.get("url", "").strip()

                    search_result.append(
                        {
                            "snippet": snippet,
                            "title": title,
                            "url": url
                        }
                    )

                return search_result

            except Exception as e:
                self.logger.error(
                    "反序列化 MCP 结果失败: %s",
                    e
                )
                return []

        finally:
            # 无论成功、异常还是 Timeout，
            # 都尽量关闭 MCP 连接
            try:
                await mcp_client.cleanup()
            except Exception as e:
                self.logger.warning(
                    "MCP Client cleanup 失败: %s",
                    e
                )


if __name__ == "__main__":
    state = {
        "rewritten_query": "今天的小米汽车的股价是多少",
        "item_names": ["数字万用表"]
    }

    mcp_search = McpSearchNode()

    result = mcp_search.process(state)

    for r in result.get("web_search_docs", []):
        print(
            json.dumps(
                r,
                ensure_ascii=False,
                indent=2
            )
        )