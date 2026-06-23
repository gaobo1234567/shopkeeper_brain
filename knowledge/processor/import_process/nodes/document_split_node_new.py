import json
import os
import re
from typing import Any, Dict, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.state import ImportGraphState


class DocumentSplitNode(BaseNode):
    name = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        md_content, file_title, max_content_length, min_content_length = self._get_inputs(state)
        sections = self._split_by_headings(md_content, file_title)
        final_chunks = self.split_and_merge(sections, max_content_length, min_content_length)
        chunks = self._assemble_chunk(final_chunks)

        state["chunks"] = chunks
        self._log_summary(md_content, chunks, max_content_length)
        self._backup_chunks(state, chunks)
        return state

    def _get_inputs(self, state: ImportGraphState) -> Tuple[str, str, int, int]:
        self.log_step("step1", "validate split inputs")
        config = get_config()

        md_content = state.get("md_content") or ""
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        file_title = state.get("file_title") or ""

        if config.max_content_length <= 0 or config.min_content_length <= 0:
            raise ValueError("invalid chunk length config")
        if config.max_content_length <= config.min_content_length:
            raise ValueError("max_content_length must be greater than min_content_length")

        return md_content, file_title, config.max_content_length, config.min_content_length

    def _split_by_headings(self, md_content: str, file_title: str) -> List[Dict[str, Any]]:
        self.log_step("step2", "split by markdown headings")

        in_fence = False
        body_lines: List[str] = []
        sections: List[Dict[str, Any]] = []
        current_level = 0
        current_title = ""
        hierarchy = [""] * 7
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")

        def flush_section() -> None:
            nonlocal body_lines

            body = "\n".join(body_lines)
            if not current_title and not body:
                return

            parent_title = ""
            for index in range(current_level - 1, 0, -1):
                if hierarchy[index]:
                    parent_title = hierarchy[index]
                    break

            if not parent_title:
                parent_title = current_title if current_title else file_title

            sections.append({
                "title": current_title if current_title else file_title,
                "body": body,
                "file_title": file_title,
                "parent_title": parent_title,
            })

        for line in md_content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_fence = not in_fence

            match = heading_re.match(line) if not in_fence else None
            if match:
                flush_section()
                current_level = len(match.group(1))
                current_title = line
                hierarchy[current_level] = current_title
                for index in range(current_level + 1, 7):
                    hierarchy[index] = ""
                body_lines = []
            else:
                body_lines.append(line)

        flush_section()
        return sections

    def split_and_merge(
        self,
        sections: List[Dict[str, Any]],
        max_content_length: int,
        min_content_length: int,
    ) -> List[Dict[str, Any]]:
        self.log_step("step3", "split long sections and merge short sections")

        current_sections: List[Dict[str, Any]] = []
        for section in sections:
            current_sections.extend(self.split_long_section(section, max_content_length))

        return self.merge_short_section(current_sections, min_content_length)

    def split_long_section(
        self,
        section: Dict[str, Any],
        max_content_length: int,
    ) -> List[Dict[str, Any]]:
        title = section.get("title") or ""
        body = section.get("body") or ""
        file_title = section.get("file_title") or ""
        parent_title = section.get("parent_title") or ""

        title_max_length = 50
        if len(title) > title_max_length:
            self.logger.warning("section title too long: %s / %s", file_title, title)
            title = title[:title_max_length]

        title_prefix = f"{title}\n\n"
        total_length = len(title_prefix) + len(body)
        if total_length <= max_content_length:
            return [section]

        body_length = max_content_length - len(title_prefix)
        if body_length <= 0:
            return [section]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_length,
            chunk_overlap=0,
            separators=[
                "\n\n",
                "\n",
                "\u3002",
                "\uff0c",
                "\uff1b",
                "\uff1a",
                ".",
                "!",
                "?",
                ";",
                " ",
                "",
            ],
            keep_separator=False,
        )
        texts = text_splitter.split_text(body)
        if len(texts) <= 1:
            return [section]

        return [
            {
                "title": f"{title}-{index + 1}",
                "body": text,
                "file_title": file_title,
                "parent_title": parent_title,
                "part": f"{index + 1}",
            }
            for index, text in enumerate(texts)
        ]

    def merge_short_section(
        self,
        current_sections: List[Dict[str, Any]],
        min_content_length: int,
    ) -> List[Dict[str, Any]]:
        if not current_sections:
            return []

        current_section = current_sections[0]
        final_sections: List[Dict[str, Any]] = []

        for next_section in current_sections[1:]:
            same_parent = current_section.get("parent_title") == next_section.get("parent_title")
            current_body = current_section.get("body") or ""

            if same_parent and len(current_body) < min_content_length:
                current_section["body"] = (
                    current_body.rstrip() + "\n\n" + (next_section.get("body") or "").lstrip()
                )
                current_section["title"] = current_section.get("parent_title")
                current_section["part"] = 0
            else:
                final_sections.append(current_section)
                current_section = next_section

        final_sections.append(current_section)

        part_counter: Dict[str, int] = {}
        result: List[Dict[str, Any]] = []
        for section in final_sections:
            if "part" in section:
                parent_title = section.get("parent_title") or ""
                part_counter[parent_title] = part_counter.get(parent_title, 0) + 1
                new_part = part_counter[parent_title]
                section["part"] = new_part
                section["title"] = f"{section.get('title')}- {new_part}"

            result.append(section)

        return result

    def _assemble_chunk(self, final_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.log_step("step4", "assemble chunks")

        chunks = []
        for chunk in final_chunks:
            title = chunk.get("title")
            body = chunk.get("body") or ""
            assembled_chunk = {
                "title": title,
                "file_title": chunk.get("file_title"),
                "parent_title": chunk.get("parent_title"),
                "content": f"{title}\n\n{body}",
            }
            if "part" in chunk:
                assembled_chunk["part"] = chunk.get("part")
            chunks.append(assembled_chunk)

        return chunks

    def _log_summary(self, raw_content: str, chunks: List[Dict[str, Any]], max_length: int) -> None:
        self.log_step("step5", "split summary")
        self.logger.info("raw lines: %s", raw_content.count("\n") + 1)
        self.logger.info("final chunks: %s", len(chunks))
        self.logger.info("max chunk length: %s", max_length)

    def _backup_chunks(self, state: ImportGraphState, chunks: List[Dict[str, Any]]) -> None:
        self.log_step("step6", "backup chunks")

        local_dir = state.get("file_dir", "")
        if not local_dir:
            self.logger.debug("file_dir is empty, skip chunks backup")
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(chunks, file, ensure_ascii=False, indent=2)
            self.logger.info("chunks backed up to: %s", output_path)
        except Exception as exc:
            self.logger.warning("chunks backup failed: %s", exc)


DocumentSplitNodeNew = DocumentSplitNode


if __name__ == "__main__":
    setup_logging()
