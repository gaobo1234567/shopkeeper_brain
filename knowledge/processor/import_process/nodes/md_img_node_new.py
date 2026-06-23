import base64
import re
import time
from collections import deque
from dataclasses import dataclass
from html import unescape
from pathlib import Path, PurePosixPath
from typing import Deque
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import (
    FileProcessingError,
    ImageProcessingError,
    ValidationError,
)
from knowledge.processor.import_process.state import ImportGraphState
from urllib.parse import unquote, urlsplit

from knowledge.utils.markdown_util import MarkdownTableLinearizer
from knowledge.utils.minio_util import get_minio_client


@dataclass
class ImageReference:
    key: str
    kind: str
    raw: str
    src: str
    image_name: str
    image_path: Path
    start: int
    end: int
    context: Tuple[str, str, str]


class MarkDownImageNode(BaseNode):
    """
    Optimized Markdown image node.

    Improvements:
    - Supports Markdown images and HTML <img> tags.
    - Matches local image references by exact normalized basename instead of
      substring contains matching.
    - Uses character spans to locate image context, so images embedded in
      paragraphs, tables, or inline HTML can be processed.
    - Linearizes HTML/Markdown tables before document splitting.
    """

    name = "md_img_node"

    MARKDOWN_IMAGE_RE = re.compile(
        r"!\[([^\]]*)\]\(\s*([^)\s]+)(?:\s+['\"][^)]*['\"])?\s*\)",
        re.IGNORECASE | re.DOTALL,
    )
    HTML_IMG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE | re.DOTALL)
    HTML_SRC_RE = re.compile(
        r"""\bsrc\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""",
        re.IGNORECASE | re.DOTALL,
    )
    HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+.+$")

    def process(self, state: ImportGraphState) -> ImportGraphState:
        config = get_config()
        md_content, md_path_obj, image_dir = self._get_img_md_content(state)

        if image_dir.exists():
            image_refs = self._collect_image_references(image_dir, md_content, config)
            image_summaries = self._extract_reference_summaries(
                md_path_obj.stem, image_refs, config
            )
            new_md_content = self._upload_and_replace_references(
                md_path_obj.stem, md_content, image_refs, image_summaries, config
            )
        else:
            self.logger.warning(f"文件{md_path_obj.name}暂无本地图片目录，跳过图片摘要处理")
            new_md_content = md_content

        new_md_content = self._linearize_tables(new_md_content)
        self._backup_new_md_file(md_path_obj, new_md_content)
        state["md_content"] = new_md_content
        return state

    def _backup_new_md_file(self, md_path_obj: Path, new_md_content: str) -> str:
        self.log_step("step5", "backup processed markdown")

        new_file_path = md_path_obj.with_name(f"{md_path_obj.stem}_new{md_path_obj.suffix}")
        try:
            with open(new_file_path, "w", encoding="utf-8") as file:
                file.write(new_md_content)
            self.logger.info("processed markdown backed up to: %s", new_file_path)
        except IOError as exc:
            self.logger.error("processed markdown backup failed: %s", exc)
            raise ImageProcessingError(f"markdown backup failed: {exc}", node_name=self.name)

        return str(new_file_path)

    def _get_img_md_content(self, state: ImportGraphState):
        md_path = state.get("md_path", "")
        if not md_path:
            raise ValidationError("md文件不存在", self.name)

        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise FileProcessingError("md文件路径无效", self.name)

        with open(md_path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()

        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        return md_content, md_path_obj, md_path_obj.parent / "images"

    def _collect_image_references(self, image_dir: Path, md_content: str, config) -> List[ImageReference]:
        self.log_step("step2", f"扫描图片引用: {image_dir}")

        image_files = self._build_image_file_index(image_dir, config)
        if not image_files:
            self.logger.info("图片目录中没有可处理的图片文件")
            return []

        refs: List[ImageReference] = []
        occupied_spans: List[Tuple[int, int]] = []

        for match in self.MARKDOWN_IMAGE_RE.finditer(md_content):
            src = match.group(2)
            image_name = self._image_name_from_src(src)
            image_path = image_files.get(image_name.casefold()) if image_name else None
            if not image_path:
                continue

            ref = self._build_image_reference(
                kind="markdown",
                raw=match.group(0),
                src=src,
                image_name=image_path.name,
                image_path=image_path,
                start=match.start(),
                end=match.end(),
                md_content=md_content,
                max_chars=config.img_content_length,
                index=len(refs),
            )
            refs.append(ref)
            occupied_spans.append((match.start(), match.end()))

        for match in self.HTML_IMG_RE.finditer(md_content):
            if self._span_overlaps(match.start(), match.end(), occupied_spans):
                continue

            src = self._extract_html_img_src(match.group(0))
            image_name = self._image_name_from_src(src) if src else ""
            image_path = image_files.get(image_name.casefold()) if image_name else None
            if not image_path:
                continue

            refs.append(self._build_image_reference(
                kind="html",
                raw=match.group(0),
                src=src,
                image_name=image_path.name,
                image_path=image_path,
                start=match.start(),
                end=match.end(),
                md_content=md_content,
                max_chars=config.img_content_length,
                index=len(refs),
            ))

        refs.sort(key=lambda item: item.start)
        self.logger.info(f"找到{len(refs)}个有效图片引用")
        return refs

    def _build_image_file_index(self, image_dir: Path, config) -> Dict[str, Path]:
        image_files: Dict[str, Path] = {}
        valid_exts = {ext.lower() for ext in config.image_extensions}
        for child in image_dir.iterdir():
            if not child.is_file() or child.suffix.lower() not in valid_exts:
                continue
            image_files[child.name.casefold()] = child
        return image_files

    def _build_image_reference(
        self,
        kind: str,
        raw: str,
        src: str,
        image_name: str,
        image_path: Path,
        start: int,
        end: int,
        md_content: str,
        max_chars: int,
        index: int,
    ) -> ImageReference:
        return ImageReference(
            key=f"{image_name}#{index}",
            kind=kind,
            raw=raw,
            src=src,
            image_name=image_name,
            image_path=image_path,
            start=start,
            end=end,
            context=self._find_context_by_span(md_content, start, end, max_chars),
        )

    def _image_name_from_src(self, src: Optional[str]) -> str:
        if not src:
            return ""

        src = unescape(src.strip().strip('"').strip("'"))
        lowered = src.lower()
        if lowered.startswith(("http://", "https://", "data:", "blob:")):
            return ""

        split_result = urlsplit(src)
        path = unquote(split_result.path or src)
        path = path.replace("\\", "/")
        return PurePosixPath(path).name

    def _extract_html_img_src(self, img_tag: str) -> str:
        match = self.HTML_SRC_RE.search(img_tag)
        if not match:
            return ""
        return next((group for group in match.groups() if group), "")

    def _span_overlaps(self, start: int, end: int, spans: List[Tuple[int, int]]) -> bool:
        return any(start < span_end and end > span_start for span_start, span_end in spans)

    def _find_context_by_span(
        self, md_content: str, start: int, end: int, max_chars: int
    ) -> Tuple[str, str, str]:
        headings = list(self.HEADING_RE.finditer(md_content))
        previous_heading = None
        next_heading = None

        for heading in headings:
            if heading.start() < start:
                previous_heading = heading
                continue
            if heading.start() > end:
                next_heading = heading
                break

        section_title = previous_heading.group(0).strip() if previous_heading else ""
        section_start = previous_heading.end() if previous_heading else 0
        section_end = next_heading.start() if next_heading else len(md_content)

        pre_content = md_content[section_start:start]
        post_content = md_content[end:section_end]

        return (
            section_title,
            self._extract_context_from_text(pre_content, max_chars, reverse=True),
            self._extract_context_from_text(post_content, max_chars, reverse=False),
        )

    def _extract_context_from_text(self, content: str, max_chars: int, reverse: bool) -> str:
        content = self.MARKDOWN_IMAGE_RE.sub("", content)
        content = self.HTML_IMG_RE.sub("", content)

        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", content)
            if paragraph.strip()
        ]
        if reverse:
            paragraphs.reverse()

        selected = []
        total = 0
        for paragraph in paragraphs:
            paragraph_len = len(paragraph)
            if total + paragraph_len > max_chars and selected:
                break
            selected.append(paragraph)
            total += paragraph_len

        if reverse:
            selected.reverse()

        return "\n\n".join(selected)

    def _extract_img_summary(
        self,
        document_title: str,
        target_images_context: List[Tuple[str, str, Tuple[str, str, str]]],
        config,
    ) -> Dict[str, str]:
        self.log_step("step3", "extract image summaries")

        summaries: Dict[str, str] = {}
        request_timestamps: Deque[float] = deque()

        try:
            client = OpenAI(api_key=config.openai_api_key, base_url=config.openai_api_base)
        except Exception as exc:
            self.logger.error("VLM client init failed: %s", exc)
            return summaries

        for image_key, image_path, image_context in target_images_context:
            self._enforce_rate_limit(request_timestamps, config.requests_per_minute, 60)
            summaries[image_key] = self._get_img_summary(
                config,
                client,
                document_title,
                image_path,
                image_context,
            )

        self.logger.info("generated image summaries: %s", len(summaries))
        return summaries

    def _enforce_rate_limit(
        self,
        request_timestamps: Deque[float],
        max_requests: int,
        window_seconds: int = 60,
    ) -> None:
        if max_requests <= 0:
            return

        current_time = time.time()
        while request_timestamps and current_time - request_timestamps[0] >= window_seconds:
            request_timestamps.popleft()

        if len(request_timestamps) >= max_requests:
            sleep_duration = window_seconds - (current_time - request_timestamps[0])
            if sleep_duration > 0:
                self.logger.info("rate limit reached, sleep %.2f seconds", sleep_duration)
                time.sleep(sleep_duration)

            current_time = time.time()
            while request_timestamps and current_time - request_timestamps[0] >= window_seconds:
                request_timestamps.popleft()

        request_timestamps.append(time.time())

    def _get_img_summary(
        self,
        config,
        client: OpenAI,
        document_title: str,
        img_path: str,
        image_context: Tuple[str, str, str],
    ) -> str:
        section_title, pre_context, post_context = image_context

        context_parts = []
        if section_title:
            context_parts.append(section_title)
        if pre_context:
            context_parts.append(pre_context)
        if post_context:
            context_parts.append(post_context)

        final_context = "\n".join(context_parts) if context_parts else "no context"

        try:
            with open(img_path, "rb") as file:
                local_img_content = base64.b64encode(file.read()).decode("utf-8")
        except Exception as exc:
            self.logger.warning("read image failed: %s / %s", img_path, exc)
            return "image description"

        try:
            response = client.chat.completions.create(
                model=config.vl_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Generate a concise Chinese title for this Markdown image. "
                                    f"Document title: {document_title}\n"
                                    f"Context: {final_context}\n"
                                    "Only output the title, without explanation."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{local_img_content}"
                                },
                            },
                        ],
                    }
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            self.logger.warning("image summary generation failed: %s / %s", img_path, exc)
            return "image description"

    def _extract_reference_summaries(
        self, document_title: str, image_refs: List[ImageReference], config
    ) -> Dict[str, str]:
        targets = [
            (ref.key, str(ref.image_path), ref.context)
            for ref in image_refs
        ]
        return self._extract_img_summary(document_title, targets, config)

    def _upload_and_replace_references(
        self,
        document_name: str,
        md_content: str,
        image_refs: List[ImageReference],
        image_summaries: Dict[str, str],
        config,
    ) -> str:
        if not image_refs:
            return md_content

        remote_urls = self._upload_unique_images(document_name, image_refs, config)

        replacements = []
        for ref in image_refs:
            remote_url = remote_urls.get(ref.image_name)
            if not remote_url:
                continue

            summary = image_summaries.get(ref.key) or "图片描述"
            summary = self._clean_markdown_alt_text(summary)
            replacements.append((ref.start, ref.end, f"![{summary}]({remote_url})"))

        new_md_content = md_content
        for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
            new_md_content = new_md_content[:start] + replacement + new_md_content[end:]

        return new_md_content

    def _upload_unique_images(
        self, document_name: str, image_refs: List[ImageReference], config
    ) -> Dict[str, str]:
        remote_urls: Dict[str, str] = {}
        minio_client = get_minio_client()

        unique_refs: Dict[str, ImageReference] = {}
        for ref in image_refs:
            unique_refs.setdefault(ref.image_name, ref)

        for image_name, ref in unique_refs.items():
            object_name = f"{document_name}/{image_name}"

            if minio_client is None:
                self.logger.warning(f"无法上传{image_name}到minio，使用mock地址")
                remote_urls[image_name] = f"http://minio_mock/{document_name}/{image_name}"
                continue

            try:
                minio_client.fput_object(config.minio_bucket, object_name, str(ref.image_path))
                remote_urls[image_name] = (
                    config.get_minio_base_url() + "/" + config.minio_bucket + "/" + object_name
                )
                self.logger.info(f"{image_name}图片上传到minio成功")
            except Exception as exc:
                self.logger.warning(f"{image_name}上传到minio失败: {exc}")
                remote_urls[image_name] = f"http://minio_mock/{document_name}/{image_name}"

        self.logger.info(f"成功处理{len(remote_urls)}张唯一图片的远程地址")
        return remote_urls

    def _clean_markdown_alt_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        return text.replace("[", "").replace("]", "") or "图片描述"

    def _linearize_tables(self, md_content: str) -> str:
        self.log_step("step_table", "提前线性化表格")
        return MarkdownTableLinearizer.process(md_content)


MarkDownImageNodeNew = MarkDownImageNode


if __name__ == "__main__":
    setup_logging()
    img_md_node = MarkDownImageNode()
    state = {
        "md_path": r"D:\pycharmprojects\shopkeeper_brain\knowledge\processor\import_process\output_temp_dir\万用表的使用\hybrid_auto\万用表的使用.md"
    }
    img_md_node.process(state)
