"""查询流程节点基类

定义统一的节点接口规范，提供通用功能。
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Optional
import logging

from knowledge.processor.query_process.config import QueryConfig, get_config
from knowledge.processor.query_process.exceptions import QueryProcessError
from knowledge.utils.sse_util import push_sse_event
from knowledge.utils.task_util import add_running_task, add_done_task, get_done_task_list, get_running_task_list, \
    get_task_status
import time
T = TypeVar("T")  # 泛型状态类型


class BaseNode(ABC):
    """查询流程节点基类。

    所有节点类都应继承此基类，实现 process 方法。
    基类提供统一的日志、任务追踪和错误处理。

    Attributes:
        name: 节点名称，子类应覆盖。
        config: 配置对象。
        logger: 日志记录器。

    Example:
        >>> class MyNode(BaseNode):
        ...     name = "my_node"
        ...
        ...     def process(self, state):
        ...         # 实现具体逻辑
        ...         return state
        ...
        >>> # 作为 LangGraph 节点使用
        >>> node = MyNode()
        >>> workflow.add_node("my_node", node)
    """

    name: str = "base_node"

    def __init__(self, config: Optional[QueryConfig] = None):
        """初始化节点。

        Args:
            config: 配置对象，默认使用全局配置。
        """
        self.config = config or get_config()
        self.logger = logging.getLogger(f"query.{self.name}")

    def __call__(self, state: T) -> T:

        task_id = state.get("task_id", "")
        is_stream = state.get("is_stream", False)

        start_time = time.perf_counter()

        try:
            self.logger.info(
                "node=%s task_id=%s status=start",
                self.name,
                task_id
            )

            if task_id:
                add_running_task(task_id, self.name)

                if is_stream:
                    self._push_progress(task_id)

            result = self.process(state)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            self.logger.info(
                "node=%s task_id=%s status=success latency_ms=%.2f",
                self.name,
                task_id,
                elapsed_ms
            )

            if task_id:
                add_done_task(task_id, self.name)

                if is_stream:
                    self._push_progress(task_id)

            return result

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            self.logger.error(
                "node=%s task_id=%s status=failed latency_ms=%.2f error=%s",
                self.name,
                task_id,
                elapsed_ms,
                e
            )

            raise QueryProcessError(
                message=str(e),
                node_name=self.name,
                cause=e
            )

    @staticmethod
    def _push_progress(task_id: str):
        push_sse_event(task_id, "progress", {
            "status": get_task_status(task_id),
            "done_list": get_done_task_list(task_id),
            "running_list": get_running_task_list(task_id),
        })

    @abstractmethod
    def process(self, state: T) -> T:
        """节点核心处理逻辑。

        子类必须实现此方法。

        Args:
            state: 图状态字典。

        Returns:
            更新后的状态字典。
        """
        pass

    def log_step(self, step_name: str, message: str = ""):
        """记录步骤日志。

        Args:
            step_name: 步骤名称。
            message: 附加信息。
        """
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)


def setup_logging(level: int = logging.INFO):
    """配置查询流程日志。

    Args:
        level: 日志级别，默认 INFO。
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
