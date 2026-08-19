import os, re, json
from typing import Tuple, List, Dict, Any
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    name = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 加载--->打散（1. 嵌入模型语义更准确 2.注入元数据 3.多路召回 4.性能、成本低
        # ----->减少LLM的幻觉，提高检索质量）---->组合:LLM就想成人的脑子

        # 1. 获取参数
        md_content, file_title, max_content_length, min_content_length = self._get_inputs(state)

        # 2. 根据标题切割(核心)
        sections = self._split_by_headings(md_content, file_title)

        # 3. 处理(切分和合并)
        final_chunks = self.split_and_merge(sections, max_content_length, min_content_length)

        # 4. 组装
        chunks = self._assemble_chunk(final_chunks)

        # 5. 更新state:chunks
        state['chunks'] = chunks

        # 6. 日志统计
        self._log_summary(md_content, chunks, max_content_length)

        # 7. 备份
        state["chunks"] = chunks
        self._backup_chunks(state, chunks)

        # 8. 返回
        return state

    # ------------------------------------------------------------------ #
    #                       日志 & 备份                                    #
    # ------------------------------------------------------------------ #

    def _log_summary(self, raw_content: str, chunks: List[dict], max_length: int):
        """输出切分统计信息"""
        self.log_step("step5", "输出统计")

        lines_count = raw_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(chunks)}")
        self.logger.info(f"最大切片长度: {max_length}")

        if chunks:
            self.logger.info("章节预览:")
            for i, sec in enumerate(chunks[:5]):
                title = sec.get("title", "")[:30]
                self.logger.info(f"  {i + 1}. {title}...")
            if len(chunks) > 5:
                self.logger.info(f"  ... 还有 {len(chunks) - 5} 个章节")

    def _backup_chunks(self, state: ImportGraphState, sections: List[dict]):
        """将切分结果备份到 JSON 文件"""
        self.log_step("step6", "备份切片")

        local_dir = state.get("file_dir", "")
        if not local_dir:
            self.logger.debug("未设置 file_dir，跳过备份")
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")

        except Exception as e:
            self.logger.warning(f"备份失败: {e}")

    def _get_inputs(self, state: ImportGraphState) -> Tuple[str, str, int, int]:
        self.log_step("step1", "切分文档的参数校验以及获取...")

        config = get_config()
        # 1. 获取md_content
        md_content = state.get('md_content')

        # 2. 统一换行符
        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        # 3. 获取文件标题
        file_title = state.get('file_title')

        # 4. 校验最大最小值
        if config.max_content_length <= 0 or config.min_content_length <= 0 or config.max_content_length <= config.min_content_length:
            raise ValueError("切片长度参数校验失败")
        return md_content, file_title, config.max_content_length, config.min_content_length

    def _split_by_headings(self, md_content: str, file_title: str) -> List[dict]:
        """
        根据MD的标题（1-6）级标题进行切分
        """
        self.log_step("step2", "根据标题进行切分...")
        # 1. 定义变量
        in_fence = False
        body_lines = []
        sections = []
        current_level = 0
        current_title = ""
        hierarchy = [""] * 7  # 7个长度 但是第一个（0）不用

        # 2. 定义正则表达式(group1:标题的语法符号#【最少1个# 最多6个#】)
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")

        # 3. 切分
        content_lines = md_content.split("\n")

        def _flush():
            body = "\n".join(body_lines)
            if current_title or body:
                parent_title = ""
                for i in range(current_level - 1, 0, -1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break

                if not parent_title:
                    parent_title = current_title if current_title else file_title

                sections.append({
                    "title": current_title if current_title else file_title,
                    "body": body,
                    "file_title": file_title,
                    "parent_title": parent_title
                })

        for content_line in content_lines:
            # 3.1 判断是否存在代码块围栏（python的注释就是#）
            if content_line.strip().startswith("```") or content_line.strip().startswith("~~~"):
                in_fence = not in_fence

            match = heading_re.match(content_line) if not in_fence else None

            if match:
                _flush()

                level = len(match.group(1))
                current_level = level
                current_title = content_line
                hierarchy[level] = current_title

                for i in range(level + 1, 7):
                    hierarchy[i] = ""

                body_lines = []
            else:
                body_lines.append(content_line)

        _flush()
        return sections

    def split_and_merge(self, sections: List[Dict[str, Any]], max_content_length: int, min_content_length: int):
        """
        1. 长块切分：保护图片+detail、Markdown表格、HTML表格，不从这些语义单元内部切开。
        2. 短块合并：仅在同父标题、当前块过短且合并后不超过max_content_length时合并。
        """
        self.log_step("step3", "切分及合并...")

        # 1. 切分
        current_sections = []
        for section in sections:
            current_sections.extend(self.split_long_section(section, max_content_length))

        # 2. 合并
        final_sections = self.merge_short_section(
            current_sections,
            min_content_length,
            max_content_length
        )

        # 3. 返回
        return final_sections

    # ------------------------------------------------------------------ #
    #               结构感知切分：保护图片和表格                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_markdown_table_separator(line: str) -> bool:
        """
        判断一行是否是Markdown表格的分隔行，例如：
        |---|---|
        |:---|---:|
        """
        stripped = line.strip()
        if "|" not in stripped:
            return False

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells:
            return False

        return all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)

    def _is_markdown_table_start(self, lines: List[str], index: int) -> bool:
        """当前行是表头且下一行是Markdown表格分隔线时，认为表格开始。"""
        if index + 1 >= len(lines):
            return False

        header = lines[index].strip()
        separator = lines[index + 1]
        return "|" in header and self._is_markdown_table_separator(separator)

    @staticmethod
    def _is_enriched_image_line(line: str) -> bool:
        """识别标准Markdown图片语法。"""
        return bool(re.match(r"^\s*!\[.*?\]\(.*?\)\s*$", line))

    @staticmethod
    def _is_image_detail_line(line: str) -> bool:
        """
        识别上一图片节点新增的图片语义增强文本。
        当前优化版图片节点写入形式类似：
        > 图片类型：table；图片详细内容：......
        """
        stripped = line.strip()
        return stripped.startswith(">") and (
            "图片类型：" in stripped or "图片详细内容：" in stripped
        )

    def _extract_structure_blocks(self, body: str) -> List[Dict[str, Any]]:
        """
        把section正文拆成普通文本块和“不可拆语义单元”。

        保护三类结构：
        1. Markdown图片 + 紧随其后的图片detail
        2. Markdown表格
        3. HTML <table>...</table> 表格

        返回：
        [
            {"type": "text", "text": "...", "atomic": False},
            {"type": "image", "text": "...", "atomic": True},
            {"type": "table", "text": "...", "atomic": True},
        ]
        """
        lines = body.split("\n")
        blocks: List[Dict[str, Any]] = []
        normal_lines: List[str] = []
        in_fence = False
        i = 0

        def flush_normal():
            if not normal_lines:
                return
            text = "\n".join(normal_lines).strip("\n")
            if text:
                blocks.append({"type": "text", "text": text, "atomic": False})
            normal_lines.clear()

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # fenced code block内部不识别图片/表格结构，避免把示例代码误判成真实内容
            if stripped.startswith("```") or stripped.startswith("~~~"):
                normal_lines.append(line)
                in_fence = not in_fence
                i += 1
                continue

            if in_fence:
                normal_lines.append(line)
                i += 1
                continue

            # 1. HTML表格：整张表作为一个原子单元，并沿用原项目的线性化逻辑
            if re.search(r"<table\b", line, flags=re.IGNORECASE):
                flush_normal()
                table_lines = [line]
                i += 1
                while i < len(lines):
                    table_lines.append(lines[i])
                    if re.search(r"</table\s*>", lines[i], flags=re.IGNORECASE):
                        i += 1
                        break
                    i += 1

                html_table = "\n".join(table_lines)
                try:
                    table_text = MarkdownTableLinearizer.process(html_table)
                except Exception as e:
                    self.logger.warning(f"HTML表格线性化失败，保留原始表格: {e}")
                    table_text = html_table

                blocks.append({"type": "table", "text": table_text.strip(), "atomic": True})
                continue

            # 2. Markdown表格：表头 + 分隔线 + 后续数据行整体保护
            if self._is_markdown_table_start(lines, i):
                flush_normal()
                table_lines = [lines[i], lines[i + 1]]
                i += 2

                while i < len(lines):
                    row = lines[i]
                    # Markdown表格的数据行一般仍然包含“|”；遇到空行或非表格行停止
                    if not row.strip() or "|" not in row:
                        break
                    table_lines.append(row)
                    i += 1

                blocks.append({
                    "type": "table",
                    "text": "\n".join(table_lines).strip(),
                    "atomic": True
                })
                continue

            # 3. 图片 + detail：作为同一个不可拆语义单元
            if self._is_enriched_image_line(line):
                flush_normal()
                image_lines = [line]
                j = i + 1

                # 允许图片与detail之间存在空行，但只有确实检测到detail时才把空行并入图片块
                blank_lines: List[str] = []
                while j < len(lines) and not lines[j].strip():
                    blank_lines.append(lines[j])
                    j += 1

                if j < len(lines) and self._is_image_detail_line(lines[j]):
                    image_lines.extend(blank_lines)
                    while j < len(lines) and self._is_image_detail_line(lines[j]):
                        image_lines.append(lines[j])
                        j += 1
                    i = j
                else:
                    i += 1

                blocks.append({
                    "type": "image",
                    "text": "\n".join(image_lines).strip(),
                    "atomic": True
                })
                continue

            normal_lines.append(line)
            i += 1

        flush_normal()
        return blocks

    def _split_body_preserving_structure(self, body: str, body_length: int) -> List[str]:
        """
        结构感知切分：
        - 普通文本仍然使用RecursiveCharacterTextSplitter；
        - 图片+detail、Markdown表格、HTML表格不从内部切开；
        - 再把各片段按body_length尽量装箱。
        """
        blocks = self._extract_structure_blocks(body)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_length,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
            keep_separator=False
        )

        # 第一步：普通文本可递归切，特殊结构保持原子性
        fragments: List[Dict[str, Any]] = []
        for block in blocks:
            block_text = block["text"].strip()
            if not block_text:
                continue

            if block["atomic"]:
                fragments.append(block)
                continue

            if len(block_text) <= body_length:
                fragments.append(block)
            else:
                for text in text_splitter.split_text(block_text):
                    if text.strip():
                        fragments.append({"type": "text", "text": text.strip(), "atomic": False})

        # 第二步：按长度把片段重新装箱；特殊结构自身永不被拆
        texts: List[str] = []
        current_parts: List[str] = []
        current_length = 0

        def flush_current():
            nonlocal current_parts, current_length
            if current_parts:
                texts.append("\n\n".join(current_parts).strip())
                current_parts = []
                current_length = 0

        for fragment in fragments:
            fragment_text = fragment["text"].strip()
            fragment_length = len(fragment_text)

            # 单个图片/表格本身就超过上限时，为保证结构完整，允许它单独形成超长chunk
            if fragment["atomic"] and fragment_length > body_length:
                flush_current()
                self.logger.warning(
                    f"检测到不可拆结构长度{fragment_length}超过正文上限{body_length}，"
                    f"为保持结构完整，将其单独保留。类型: {fragment['type']}"
                )
                texts.append(fragment_text)
                continue

            separator_length = 2 if current_parts else 0
            if current_parts and current_length + separator_length + fragment_length > body_length:
                flush_current()

            current_parts.append(fragment_text)
            current_length += (2 if len(current_parts) > 1 else 0) + fragment_length

        flush_current()
        return texts

    def split_long_section(self, section: Dict[str, Any], max_content_length: int):
        """
        只要满足条件的才会切（当前section的内容达到了max_content_length）。
        优化点：切长文本时保护图片+detail、Markdown表格、HTML表格。
        """

        self.log_step("step3", "进行长内容的结构感知切分")

        # 1. 获取section对象属性
        title = section.get('title')
        body = section.get('body') or ""
        file_title = section.get('file_title')
        parent_title = section.get('parent_title')

        # 2. 对标题做校验
        TITLE_MAX_LENGTH = 50
        if len(title) > TITLE_MAX_LENGTH:
            self.logger.warning(f"检测文件{file_title}对应的{title}长度过长...")
            title = title[:50]

        # 3. 拼接title前缀
        title_prefix = f"{title}\n\n"

        # 4. 计算总长度(len(title_prefix)+len(body))
        total_length = len(title_prefix) + len(body)

        # 5. 小于或者刚好满足阈值，不切
        if total_length <= max_content_length:
            return [section]

        # 6. 计算body可用长度
        body_length = max_content_length - len(title_prefix)
        if body_length <= 0:
            return [section]

        # 7. 结构感知切分：图片+detail、Markdown表格、HTML表格不从内部切开
        texts = self._split_body_preserving_structure(body, body_length)

        if len(texts) <= 1:
            return [section]

        # 8. 组装子section
        sub_section = []
        for index, text in enumerate(texts):
            sub_section.append({
                "title": title + "-" + f"{index + 1}",
                "body": text,
                "file_title": file_title,
                "parent_title": parent_title,
                "part": f"{index + 1}"
            })

        return sub_section

    def merge_short_section(
        self,
        current_sections: List[Dict[str, Any]],
        min_content_length: int,
        max_content_length: int
    ):
        """
        贪心累加算法。

        优化点：只有同时满足以下条件才合并：
        1. 同一个parent_title；
        2. 当前section正文长度小于min_content_length；
        3. 合并后的 title + body 总长度不超过max_content_length。
        """
        if not current_sections:
            return []

        # 1. 定义变量
        current_section = current_sections[0]
        final_sections = []

        # 2. 遍历以及合并
        for next_section in current_sections[1:]:
            same_parent = (current_section['parent_title'] == next_section['parent_title'])
            current_body = current_section.get('body') or ""
            next_body = next_section.get('body') or ""

            candidate_body = current_body.rstrip() + "\n\n" + next_body.lstrip()
            candidate_title = current_section['parent_title']
            candidate_total_length = len(candidate_title) + 2 + len(candidate_body)

            can_merge = (
                same_parent
                and len(current_body) < min_content_length
                and candidate_total_length <= max_content_length
            )

            if can_merge:
                current_section['body'] = candidate_body
                current_section['title'] = candidate_title
                current_section['part'] = 0
            else:
                final_sections.append(current_section)
                current_section = next_section

        # 最后一个封箱
        final_sections.append(current_section)

        # 3. 对所有section的part做处理
        part_counter = {}
        result = []
        for final_section in final_sections:
            if "part" in final_section:
                parent_title = final_section.get('parent_title')
                part_counter[parent_title] = part_counter.get(parent_title, 0) + 1
                new_part = part_counter[parent_title]
                final_section['part'] = new_part
                final_section['title'] = final_section['title'] + f"- {new_part}"

            result.append(final_section)

        return result

    def _assemble_chunk(self, final_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """最终组合chunk，用 f"{title}\n\n{body}" 生成content。"""

        self.log_step("step4", "组装最终的切片信息...")
        chunks = []
        for chunk in final_chunks:
            # 1. 获取chunk的信息
            title = chunk.get('title')
            file_title = chunk.get('file_title')
            parent_title = chunk.get('parent_title')
            body = chunk.get('body')
            content = f"{title}\n\n{body}"

            # 2. 构建最终chunk对象
            assemble_chunk = {
                "title": title,
                "file_title": file_title,
                "parent_title": parent_title,
                "content": content,
            }

            # 3.判断part是否存在
            if "part" in chunk:
                assemble_chunk['part'] = chunk.get('part')

            chunks.append(assemble_chunk)

        return chunks


if __name__ == '__main__':
    setup_logging()

    document_node = DocumentSplitNode()
    file_path = r"D:\pycharmprojects\shopkeeper_brain\knowledge\processor\import_process\output_temp_dir\万用表的使用\hybrid_auto\万用表的使用_new.md"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    state = {
        "file_title": "万用表的使用",
        "md_content": content,
        "file_dir": r"D:\pycharmprojects\shopkeeper_brain\knowledge\processor\import_process\output_temp_dir\万用表的使用\hybrid_auto"
    }
    document_node.process(state)