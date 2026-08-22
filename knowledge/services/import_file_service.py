from dotenv import load_dotenv

load_dotenv()

import os
import uuid
import json
import hashlib
import shutil
import logging

logger = logging.getLogger(__name__)
from io import BytesIO
from datetime import datetime
from typing import Tuple

from fastapi import UploadFile, HTTPException
from minio.error import S3Error

from knowledge.core.paths import get_local_base_dir
from knowledge.utils.minio_util import get_minio_client
from knowledge.services.task_service import TaskService
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.main_graph import kb_import__graph_app


class ImportFileService:
    """
    文件导入业务类

    负责：
    1. 保存上传文件到本地
    2. 计算文件 SHA256，进行内容级幂等判断
    3. 保存原始文件到 MinIO
    4. 运行文档导入 Graph
    5. Graph 完整成功后写入 MinIO 导入完成标记
    """

    def __init__(self, task_service: TaskService):
        self._task_service = task_service

    def get_date_dir(self) -> str:
        return os.path.join(
            get_local_base_dir(),
            datetime.now().strftime("%Y%m%d")
        )

    # ============================================================
    # 1. 保存上传文件到本地
    # ============================================================

    def save_upload_file_to_local(
        self,
        file: UploadFile,
        file_dir: str
    ) -> str:
        """
        将上传文件保存到本地。
        """

        # 1. 确保目录存在
        os.makedirs(file_dir, exist_ok=True)

        # 2. 构建本地路径
        import_file_path = os.path.join(
            file_dir,
            file.filename
        )

        # 3. 写入文件
        with open(import_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 4. 返回路径
        return import_file_path

    # ============================================================
    # 2. 计算文件 SHA256
    # ============================================================

    def calculate_file_sha256(
        self,
        file_path: str
    ) -> str:
        """
        根据文件二进制内容计算 SHA256。

        使用分块读取，避免大文件一次性进入内存。
        """

        sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:

            while True:

                # 每次读取 1 MB
                chunk = f.read(1024 * 1024)

                if not chunk:
                    break

                sha256.update(chunk)

        return sha256.hexdigest()

    # ============================================================
    # 3. 构建 fingerprint marker 的 MinIO Object Name
    # ============================================================

    def _get_fingerprint_object_name(
        self,
        file_sha256: str
    ) -> str:
        """
        每个成功导入的文件对应一个固定的完成标记。

        例如：
        import_fingerprints/
            a1b2c3....json
        """

        return (
            f"import_fingerprints/"
            f"{file_sha256}.json"
        )

    # ============================================================
    # 4. 判断文件是否曾经完整导入成功
    # ============================================================

    def is_file_already_imported(
        self,
        file_sha256: str
    ) -> bool:
        """
        根据 SHA256 查询 MinIO completion marker。

        marker 存在：
            说明相同内容的文件曾经完整导入成功。

        marker 不存在：
            当前文件需要继续执行完整导入流程。
        """

        minio_client = get_minio_client()

        if not minio_client:
            raise HTTPException(
                status_code=500,
                detail="MinIO 服务不可用"
            )

        bucket_name = os.getenv(
            "MINIO_BUCKET_NAME"
        )

        fingerprint_object_name = (
            self._get_fingerprint_object_name(
                file_sha256
            )
        )

        logger.info(
            "[幂等检查] 当前文件 SHA256=%s",
            file_sha256
        )
        logger.info(
            "[幂等检查] 正在查询 MinIO completion marker=%s",
            fingerprint_object_name
        )

        try:
            minio_client.stat_object(
                bucket_name,
                fingerprint_object_name
            )

            logger.info(
                "[幂等检查] 命中已有完成标记，检测到相同内容文件已经成功导入"
            )
            logger.info(
                "[幂等检查] 当前文件跳过重复 ImportGraph"
            )
            return True

        except S3Error as e:
            if e.code in (
                "NoSuchKey",
                "NoSuchObject",
                "XMinioInvalidObjectName"
            ):
                logger.info(
                    "[幂等检查] 未发现对应完成标记，当前文件需要执行完整 ImportGraph"
                )
                return False

            logger.error(
                "[幂等检查] MinIO 查询异常 code=%s error=%s",
                e.code,
                e
            )
            raise

    # ============================================================
    # 5. 上传原始文件到 MinIO
    # ============================================================

    def save_upload_file_to_minio(
        self,
        import_file_path: str,
        file: UploadFile
    ):
        """
        将用户上传的原始文件保存到 MinIO。
        """

        # 1. 获取 MinIO Client
        minio_client = get_minio_client()

        if not minio_client:
            raise HTTPException(
                status_code=500,
                detail="MinIO 服务不可用"
            )

        # 2. 构建 Object Name
        minio_object_name = (
            f"origin_files/"
            f"{datetime.now().strftime('%Y%d%m')}/"
            f"{file.filename}"
        )

        # 3. Bucket
        bucket_name = os.getenv(
            "MINIO_BUCKET_NAME"
        )

        # 4. 上传
        try:

            minio_client.fput_object(
                bucket_name,
                minio_object_name,
                import_file_path
            )

        except Exception as e:

            raise ValueError(
                f"{file.filename}文件上传失败 原因:{e}"
            )

    # ============================================================
    # 6. 写入“完整导入成功” marker
    # ============================================================

    def save_import_completion_marker(
        self,
        file_sha256: str,
        original_filename: str,
        task_id: str
    ):
        """
        只有 ImportGraph 完整执行成功之后，
        才写入该 marker。

        注意：
        原始文件成功上传到 MinIO
        不代表知识库导入成功。

        因此不能提前写 marker。
        """

        minio_client = get_minio_client()

        if not minio_client:
            raise RuntimeError(
                "MinIO 服务不可用，无法写入导入完成标记"
            )

        bucket_name = os.getenv(
            "MINIO_BUCKET_NAME"
        )

        fingerprint_object_name = (
            self._get_fingerprint_object_name(
                file_sha256
            )
        )

        marker_data = {
            "sha256": file_sha256,
            "original_filename": original_filename,
            "task_id": task_id,
            "status": "completed",
            "completed_at": datetime.now().isoformat()
        }

        marker_bytes = json.dumps(
            marker_data,
            ensure_ascii=False
        ).encode("utf-8")

        marker_stream = BytesIO(marker_bytes)

        minio_client.put_object(
            bucket_name,
            fingerprint_object_name,
            marker_stream,
            length=len(marker_bytes),
            content_type="application/json"
        )

        logger.info(
            "[幂等标记] completion marker 写入成功 object=%s",
            fingerprint_object_name
        )

    # ============================================================
    # 7. 上传文件入口
    # ============================================================

    def process_upload_file(
        self,
        file: UploadFile
    ) -> Tuple[str, str, str, str, bool]:
        """
        上传文件并进行幂等检查。

        Returns:
            task_id
            file_dir
            import_file_path
            file_sha256
            is_duplicate
        """

        # 1. 创建日期目录
        date_dir = self.get_date_dir()

        # 2. 创建任务 ID
        task_id = str(uuid.uuid4())

        # 3. 创建本次任务目录
        file_dir = os.path.join(
            date_dir,
            task_id
        )

        # 4. 标记 upload_file 节点开始
        self._task_service.mark_node_running(
            task_id,
            "upload_file"
        )

        # 5. 先把文件落到本地
        import_file_path = (
            self.save_upload_file_to_local(
                file,
                file_dir
            )
        )

        # 6. 计算文件内容 SHA256
        file_sha256 = (
            self.calculate_file_sha256(
                import_file_path
            )
        )

        logger.info(
            "[文件指纹] filename=%s sha256=%s",
            file.filename,
            file_sha256
        )

        # 7. 查询该内容以前是否完整导入成功
        is_duplicate = (
            self.is_file_already_imported(
                file_sha256
            )
        )

        # ====================================================
        # 重复文件
        # ====================================================

        if is_duplicate:

            logger.info(
                "[幂等结果] filename=%s status=duplicate action=skip",
                file.filename
            )

            self._task_service.mark_node_done(
                task_id,
                "upload_file"
            )

            self._task_service.update_task_status(
                task_id,
                "completed"
            )

            return (
                task_id,
                file_dir,
                import_file_path,
                file_sha256,
                True
            )

        # ====================================================
        # 新文件
        # ====================================================

        logger.info(
            "[幂等结果] filename=%s status=new action=continue_import",
            file.filename
        )

        # 8. 上传原始文件到 MinIO
        self.save_upload_file_to_minio(
            import_file_path,
            file
        )

        # 9. upload_file 完成
        self._task_service.mark_node_done(
            task_id,
            "upload_file"
        )

        return (
            task_id,
            file_dir,
            import_file_path,
            file_sha256,
            False
        )

    # ============================================================
    # 8. 运行完整 ImportGraph
    # ============================================================

    def run_import_graph(
        self,
        task_id: str,
        file_dir: str,
        import_file_path: str,
        file_sha256: str,
        original_filename: str
    ):
        """
        执行完整文档导入流程。

        Graph 全部成功：
            1. 写 completion marker
            2. task status = completed

        Graph 任意节点失败：
            不写 marker
            task status = failed

        因此失败任务以后可以重新上传执行。
        """

        try:

            # 1. 标记处理开始
            self._task_service.update_task_status(
                task_id,
                "processing"
            )

            # 2. 构建 LangGraph State
            global_graph_init_status: ImportGraphState = {
                "task_id": task_id,
                "file_dir": file_dir,
                "import_file_path": import_file_path
            }

            # 3. 执行整个导入 Graph
            for event in kb_import__graph_app.stream(
                global_graph_init_status
            ):

                for key, value in event.items():

                    logger.info(
                        "[%s] Completed Node: %s",
                        task_id,
                        key
                    )

            # ====================================================
            # 4. Graph 全部成功
            #    这时候才真正记录 fingerprint
            # ====================================================

            self.save_import_completion_marker(
                file_sha256=file_sha256,
                original_filename=original_filename,
                task_id=task_id
            )

            # 5. 标记任务完成
            self._task_service.update_task_status(
                task_id,
                "completed"
            )

        except Exception as e:

            # Graph 失败，不会写 completion marker
            # 所以未来可以重新上传同一个文件
            self._task_service.update_task_status(
                task_id,
                "failed"
            )

            logger.exception(
                "[%s] ImportGraph 执行失败: %s",
                task_id,
                e
            )