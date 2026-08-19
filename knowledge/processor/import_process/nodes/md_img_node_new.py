import logging
import time
import os, re
import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from pathlib import Path
from typing import Tuple, List, Deque, Dict, Any
from openai import OpenAI
from knowledge.utils.minio_util import get_minio_client
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError, FileProcessingError, ImageProcessingError
from knowledge.processor.import_process.config import get_config


class MarkDownImageNode(BaseNode):
    """
    处理MarkDown图片节点类
    """
    name = "md_img_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """

        Args:
            state: 上一个节点处理之后state的最新状态
        Returns:   当前节点处理之后的state最新状态(md_content=process_md)

        """
        # 1. 获取配置对象
        config = get_config()

        # 2. 处理文件路径(2.1 md内容 2.2 md的path 2.3 图片目录)
        md_content, md_path_obj, image_dir = self._get_img_md_content(state)
        if not image_dir.exists():
            # 图片不用处理了，直接更新state的md_content
            self.logger.warning(f"文件{md_path_obj.name}暂无图片要处理")
            state['md_content'] = md_content
            return state

        # 3. 扫描并处理图片（最复杂）
        target_images_context = self._scan_images_and_context(image_dir, md_content, config)

        # 4. 用VLM给图片生成结构化语义信息（图片类型 + 短标题 + 详细内容）
        images_summaries = self._extract_img_summary(md_path_obj.stem, target_images_context, config)

        # 5.复合函数
        # 5.1 本地图片上传到minio--->remote_url(图片远程的地址)
        # 5.2 替换md中的图片本地url，并将VLM生成的短标题/详细内容写回Markdown
        new_md_content = self._upload_img_and_update_new_md(md_path_obj.stem, md_content, images_summaries,
                                                            target_images_context, config)

        # 6. 将更新后的内容备份（调试）->new_md
        self._backup_new_md_file(md_path_obj, new_md_content)

        # 7. 更新state，所以原来的md文档不会改变，仅仅改变了状态，因此修改后的md文档在6中备份。
        state['md_content'] = new_md_content

        # 8. 返回更新后的状态
        return state

    def _backup_new_md_file(
            self,
            md_path_obj: Path,
            new_md_content: str
    ) -> str:
        self.log_step("step_5", "备份新文件")

        new_file_path = md_path_obj.with_name(
            f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        )

        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise ImageProcessingError(f"文件写入失败: {e}", node_name=self.name)

        return str(new_file_path)

    def _get_img_md_content(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        """

        Args:
            state:  上一个节点处理之后state的最新状态

        Returns:
            md_content: md的内容
            md_path_obj: md的路径
            image_dir: 图片目录
        """
        self.log_step("step1", "读取md内容以及构建图片目录")
        # 1. 获取md_path
        md_path = state.get('md_path', '')

        # 2. 判断路径是否有内容
        if not md_path:
            raise ValidationError("md文件不存在", self.name)

        # 3. 标准化处理
        md_path_obj = Path(md_path)

        # 4. 判断路径是否有效
        if not md_path_obj.exists():
            raise FileProcessingError("md文件路径无效", self.name)

        # 5. 读取md内容
        with open(md_path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()  # 全部读取

        # 6.构建图片目录
        image_dir = md_path_obj.parent / "images"

        # 7. 返回
        return md_content, md_path_obj, image_dir

    def _scan_images_and_context(self, image_dir: Path, md_content: str, config) -> List[
        Tuple[str, str, Tuple[str, str, str]]]:
        """
        扫描图片并且 找到图片的上下文
        Args:
            image_dir: 图片目录
            md_content: md内容
            config：配置信息
        Returns:
          List[Tuple[str,str,Tuple[str,str,str]]]
          List[("图片名字",“图片的地址”,("离图片最近的上面一个标题","图片的上文","图片的下文"))]
        """
        self.log_step("step2", f"扫描图片文件目录{image_dir}")
        target_images_context = []

        # 1. 遍历图片文件目录
        for img_name in os.listdir(image_dir):
            # 1.1 获取文件的后缀
            file_ext = os.path.splitext(img_name)[1]  # a.txt

            # 1.2 判断后缀是否有效
            if file_ext not in config.image_extensions:
                continue  # 继续处理下一个图片文件

            # 1.3 构建image_path 转成字符串
            img_path = str(image_dir / img_name)

            # 1.4 构建图片（上下文）
            img_context = self._find_img_context_with_limit(md_content, img_name, config.img_content_length)

            # 1.5 如果该图片没有上下文
            if not img_context:
                self.logger.warning("MD文件中暂未提取到可用的图片")
                continue  # 继续处理下一个图片文件

            # 1.6 提取到当前图片的唯一上下文内容(方便使用获取了第一个充当)
            primary_img_context = img_context[0]

            # 1.7 图片的完整信息构建到列表
            target_images_context.append((img_name, img_path, primary_img_context))

        # 1.8 返回所有图片完整信息
        self.logger.info(f"找到{len(target_images_context)}有效的图片")
        return target_images_context

    def _find_img_context_with_limit(self, md_content: str, img_name: str, max_chars=200) -> List[Tuple[str, str, str]]:
        """
         从MD文档中提取图片上下文信息
         思路：使用正则查找图片在md的位置
        Args:
            md_content:  操作的MD
            img_name:    图片名字
            max_chars:   最大的字符数限制
        Returns:
            List[Tuple[str, str, str]]
            List[("离图片最近的上面一个标题","图片的上文","图片的下文")]
        """
        # 1. 定义找图片的正则规则
        # 标准的图片在md中的语法结构：![图片描述](图片地址)
        # 正则表达式解析：
        # r:python中不要再对正则中的符号进行转义
        # ！md语法
        # [ 在正则表达式中代表字符集（a-z A-Z 0-9 + /）一共64个字符，所以要表示原来的[就需要转义
        # . 任意字符
        # * 任意字符出现的数量 0个或者多个
        # + 任意字符出现的数量 至少有一个
        # ？ 匹配模式：非贪婪模式，遇到后面的字符匹配上就停止匹配，这里是]（不加？就是贪婪模式）
        # （ 在正则表达式中代表匹配的开始和结束，这里表示匹配的开始和结束是()，这里()中的内容是匹配的内容）
        # escape对文件名做特殊处理，避免正则表达式中的特殊字符被转义

        re_pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")

        # 2. 切分md内容
        md_lines = md_content.split("\n")
        imgs_context = []

        # 3.遍历md
        for line_idx, line in enumerate(md_lines):

            # 3.1 是否是图片
            if not re_pattern.search(line):
                continue  # 继续下一行

            # 3.2 找上文标题内容和标题索引
            head_title = ""  # 初始标题内容
            head_index = -1  # 初始的标题索引
            for i in range(line_idx - 1, -1, -1):
                if re.match(r"^#{1,6}\s+", md_lines[i]):
                    head_title = md_lines[i]
                    head_index = i
                    break
            pre_content_start_index = head_index + 1
            pre_content = md_lines[pre_content_start_index:line_idx]

            # 3.3 找上文的内容(自下而上,反转)
            img_pre_context = self._extract_img_context_with_limit(pre_content, max_chars, direction="front")

            # 3.4 找下文标题索引(不要下文标题)
            section_index = len(md_lines)
            for i in range(line_idx + 1, section_index):
                if re.match(r"^#{1,6}\s+", md_lines[i]):
                    section_index = i
                    break

            post_content_start_index = line_idx + 1
            post_content = md_lines[post_content_start_index:section_index]

            # 3.5 找下文的内容(正常顺序)
            img_post_context = self._extract_img_context_with_limit(post_content, max_chars, direction="end")

            # 3.6 构建该图片的上下文信息
            imgs_context.append((head_title, img_pre_context, img_post_context))

        # 4.返回该md中当前图片的所有上下文信息（大多数情况下列表只要一个三元组对象） 除非该图片在md中有多处引用
        return imgs_context

    def _extract_img_context_with_limit(self, extract_content: list, max_chars: int, direction: str) -> str:

        """
        提取图片到上下标题（最近）之间的内容（段落）
        direction：front:自下而上
        direction：end:自上而下
        如何从给定的内容中找段落？
        策略：md中的段落 \n分割  补充：行与行之间 每一行后面都有两个空格
        Args:
            extract_content:  提取到的内容
            max_chars: 最大字符数
            direction: 方向

        Returns:
            str:上下文信息
        """
        current_paragraph = []  # 存储当前遍历到的内容
        final_paragraph = []  # 存储最终遍历到的段落（多个段落）

        # 1. 遍历每一行  收集段落
        for line in extract_content:
            #去掉字符串两边的空格，\n
            clean_strip = line.strip()
            if not clean_strip:  # 自然而然的段落分割
                if current_paragraph:
                    final_paragraph.append("\n".join(current_paragraph))
                    current_paragraph = []
            else:
                if re.match(r"^!\[.*?\]\(.*?\)$", clean_strip):  # 遇到其它图片形成的段落
                    if current_paragraph:
                        final_paragraph.append("\n".join(current_paragraph))
                        current_paragraph = []
                    continue #直接跳过图片，不会加入到上下文中
                current_paragraph.append(clean_strip)

        # 2. 处理最后一个段落且下面没有空行
        if current_paragraph:
            final_paragraph.append("\n".join(current_paragraph))

        # 3.处理上文
        if direction == "front":
            final_paragraph.reverse()  # 找打离图片最近的文档

        # 4. 收集最终返回的段落
        total = 0
        selected = []
        for para in final_paragraph:
            para_len = len(para)

            if total + para_len > max_chars and selected:
                break
            selected.append(para)  # 放入
            total += para_len  # 更新计数器

        # 5. 处理上文
        if direction == "front":
            selected.reverse()  # 返回的顺序和原文档的顺序是一致的

        # 6. 返回上下文（换行+空行）
        return "\n\n".join(selected)

    def _extract_img_summary(self, document_title: str,
                             target_images_context: List[
                                 Tuple[str, str, Tuple[str, str, str]]],
                             config) -> Dict[str, Dict[str, str]]:
        """
        并发调用VLM，为所有图片生成结构化语义信息。

        返回格式：
            {
                "图片名": {
                    "image_type": "table/chart/flowchart/screenshot/normal_image/other",
                    "caption": "简短标题",
                    "detail": "可供后续RAG检索和问答使用的详细内容"
                }
            }
        """
        self.log_step("step3", "准备并发提取图片语义信息")

        summaries: Dict[str, Dict[str, str]] = {}
        if not target_images_context:
            return summaries

        # 1. 构建OpenAI客户端（复用一个客户端，避免每张图片重复初始化）
        try:
            client = OpenAI(
                api_key=config.openai_api_key,
                base_url=config.openai_api_base
            )
        except Exception as e:
            self.logger.error(f"VLM客户端创建失败: {e}")
            return summaries

        # 2. 最大并发数：如果config暂时没有image_max_workers，则默认5
        max_workers = max(1, int(getattr(config, "image_max_workers", 5)))
        max_workers = min(max_workers, len(target_images_context))

        # 3. 所有线程共享同一个RPM时间戳队列；用Lock保证限流状态更新是原子的
        request_timestamps: Deque[float] = deque()
        rate_limit_lock = threading.Lock()

        def process_one_image(img_name: str, img_path: str, images_context: Tuple[str, str, str]):
            # 只在“申请一次请求额度”时加锁，不包住真正的网络请求，因此VLM调用仍然可以并发执行
            with rate_limit_lock:
                self._enforce_rate_limit(
                    request_timestamps,
                    config.requests_per_minute,
                    60
                )

            result = self._get_img_summary(
                config, client, document_title, img_path, images_context
            )
            return img_name, result

        # 4. 受控并发处理所有图片
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="image-vlm") as executor:
            future_to_img = {
                executor.submit(process_one_image, img_name, img_path, images_context): img_name
                for img_name, img_path, images_context in target_images_context
            }

            for future in as_completed(future_to_img):
                img_name = future_to_img[future]
                try:
                    _, result = future.result()
                    summaries[img_name] = result
                except Exception as e:
                    self.logger.warning(f"图片语义提取任务失败 {img_name}: {e}")
                    summaries[img_name] = {
                        "image_type": "unknown",
                        "caption": "",
                        "detail": ""
                    }

        # 5. 按原图片顺序重新组织，避免并发完成顺序影响后续处理/调试
        ordered_summaries = {
            img_name: summaries[img_name]
            for img_name, _, _ in target_images_context
            if img_name in summaries
        }

        self.logger.info(
            f"完成{len(ordered_summaries)}张图片的语义提取，最大并发数={max_workers}"
        )
        return ordered_summaries

    def _enforce_rate_limit(
            self,
            request_timestamps: Deque[float],
            max_requests: int,
            window_seconds: int = 60
    ):
        """
        强制执行 API 请求速率限制。

        Args:
            request_timestamps (Deque[float]): 请求时间戳队列。
            max_requests (int): 窗口内最大请求数。
            window_seconds (int, optional): 时间窗口大小（秒）。
        """
        current_time = time.time()

        # 移除窗口外的时间戳
        while request_timestamps and current_time - request_timestamps[0] >= window_seconds:
            request_timestamps.popleft()

        # 达到上限则等待
        if len(request_timestamps) >= max_requests:
            sleep_duration = window_seconds - (current_time - request_timestamps[0])
            if sleep_duration > 0:
                self.logger.info(f"达到速率限制，暂停 {sleep_duration:.2f} 秒...")
                time.sleep(sleep_duration)

            current_time = time.time()
            while request_timestamps and current_time - request_timestamps[0] >= window_seconds:
                request_timestamps.popleft()

        request_timestamps.append(current_time)

    def _get_img_summary(self, config, client, document_title: str, img_path: str,
                         images_context: Tuple[str, str, str]) -> Dict[str, str]:
        """
        调用VLM分析单张图片，返回结构化语义信息：
        image_type + caption + detail。
        """
        # 1. 解包images_context构建上下文
        section_title, pre_context, post_contex = images_context

        # 2. 判断上下文
        context_parts = []
        if section_title:
            context_parts.append(section_title)
        if pre_context:
            context_parts.append(pre_context)
        if post_contex:
            context_parts.append(post_contex)

        # 3. 构建上下文
        final_context = "\n".join(context_parts) if context_parts else "暂无可用上下文"

        # 4. 读取图片文件并转Base64
        try:
            with open(img_path, "rb") as f:
                local_img_content = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            self.logger.warning(f"图片读取失败 {img_path}: {e}")
            return {
                "image_type": "unknown",
                "caption": "",
                "detail": ""
            }

        # 5. 调用VLM：一次请求同时生成类型、短标题和详细内容
        try:
            response = client.chat.completions.create(
                model=config.vl_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""你正在为RAG知识库处理Markdown文档中的图片。
请优先依据图片本身的视觉内容判断，文档标题和上下文只作为辅助信息；如果上下文与图片视觉内容冲突，以图片本身为准，不要根据上下文臆造图片中不存在的信息。

背景信息：
1. 所属文档标题：{document_title}
2. 图片上下文：
{final_context}

请只返回一个合法JSON对象，不要使用Markdown代码块，也不要输出额外解释。JSON必须包含以下三个字段：
{{
  "image_type": "table | chart | flowchart | screenshot | normal_image | other",
  "caption": "不超过30个汉字的精准中文短标题，不要包含‘图片’二字",
  "detail": "对图片中真正可见且与问答有关的信息做详细中文描述。若是表格，尽量保留表头、行列对应关系和关键数值；若是流程图，描述关键节点和连接顺序；若是图表，描述坐标含义、关键数据和趋势；若是普通图片，描述主要对象、结构和操作信息。不要编造看不清或不存在的内容。"
}}"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{local_img_content}"
                                }
                            }
                        ]
                    }
                ]
            )

            raw_content = (response.choices[0].message.content or "").strip()
            if not raw_content:
                raise ValueError("VLM返回内容为空")

            # 6. 兼容模型偶尔返回 ```json ... ``` 的情况
            clean_content = raw_content.strip()
            if clean_content.startswith("```"):
                clean_content = re.sub(r"^```(?:json)?\s*", "", clean_content, flags=re.IGNORECASE)
                clean_content = re.sub(r"\s*```$", "", clean_content)

            try:
                parsed = json.loads(clean_content)
            except json.JSONDecodeError:
                # 如果模型没有严格遵循JSON格式，保留原始文本作为detail，避免整张图片语义完全丢失
                self.logger.warning(f"VLM返回非标准JSON {img_path}: {raw_content[:200]}")
                return {
                    "image_type": "unknown",
                    "caption": raw_content[:30],
                    "detail": raw_content
                }

            image_type = str(parsed.get("image_type", "other")).strip() or "other"
            caption = str(parsed.get("caption", "")).strip()
            detail = str(parsed.get("detail", "")).strip()

            # caption为空但detail有效时，给一个有限长度的兜底标题
            if not caption and detail:
                caption = detail[:30]

            return {
                "image_type": image_type,
                "caption": caption,
                "detail": detail
            }

        except Exception as e:
            self.logger.warning(f"图片语义生成失败 {img_path}: {e}")
            return {
                "image_type": "unknown",
                "caption": "",
                "detail": ""
            }

    def _upload_img_and_update_new_md(self, document_name, md_content, images_summaries,
                                      target_images_context, config):
        """
        上传图片到MinIO，并将图片远程URL、短标题和详细视觉内容写回Markdown。
        """
        self.log_step("step5", "上传图片到minio并且更新md的图片语义和图片地址")

        remote_urls = {}
        # 1. 构建MinIO客户端
        minio_client = get_minio_client()

        if minio_client is None:
            self.logger.warning("无法将本地的图片上传到minio")

        # 2. 遍历图片信息列表
        for img_name, img_path, _ in target_images_context:

            # 2.1 构建对象的名字
            object_name = f"{document_name}/{img_name}"

            try:
                # 2.2 开始上传
                minio_client.fput_object(
                    config.minio_bucket,
                    object_name,
                    img_path,
                )
                # 2.3 手动拼接远程地址
                remote_url = config.get_minio_base_url() + "/" + config.minio_bucket + "/" + object_name
                self.logger.info(f"{img_name}图片上传到minio成功")
                remote_urls[img_name] = remote_url

            except Exception as e:
                self.logger.warning(f"{img_name}上传到minio失败: {e}")
                remote_urls[img_name] = "http://minio_mock/" + document_name + "/" + img_name

        self.logger.info(f"成功上传{len(remote_urls)}图片到minio")

        # 3. 将图片远程URL、caption和detail写回Markdown
        new_md_content = md_content
        for img_name, image_info in images_summaries.items():
            remote_url = remote_urls.get(img_name)
            if not remote_url:
                continue

            image_type = str(image_info.get("image_type", "other")).strip() or "other"
            caption = str(image_info.get("caption", "")).strip()
            detail = str(image_info.get("detail", "")).strip()

            replace_pattern = re.compile(
                r"!\[(.*?)\]\((.*?" + re.escape(img_name) + r".*?)\)",
                re.IGNORECASE
            )

            def replace_image(match):
                # VLM失败时优先保留Markdown原本的alt，不再注入固定的“图片描述”噪声
                original_alt = (match.group(1) or "").strip()
                final_caption = caption or original_alt
                image_markdown = f"![{final_caption}]({remote_url})"

                if not detail:
                    return image_markdown

                # detail以文本形式进入Markdown，后续Chunk/Embedding才能真正检索图片内部信息
                detail_one_line = " ".join(detail.split())
                return (
                    f"{image_markdown}\n\n"
                    f"> 图片类型：{image_type}；图片详细内容：{detail_one_line}"
                )

            new_md_content = replace_pattern.sub(replace_image, new_md_content)

        return new_md_content


if __name__ == '__main__':
    setup_logging()
    img_md_node = MarkDownImageNode()

    state = {
        "md_path": r"D:\pycharmprojects\shopkeeper_brain\knowledge\processor\import_process\output_temp_dir\万用表的使用\hybrid_auto\万用表的使用.md"
    }

    img_md_node.process(state)