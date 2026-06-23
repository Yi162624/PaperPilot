"""
PDF 解析服务
提取论文文本、公式、图表、图片
"""

import re
import uuid
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF

from config import get_settings
from logger import logger

settings = get_settings()


class PDFParser:
    """PDF解析器：提取文本、公式、图表、图片"""

    def __init__(self):
        self.upload_dir = Path(settings.upload_dir)
        self.image_output_dir = Path(settings.image_output_dir)
        # 确保上传和图片输出目录存在
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.image_output_dir.mkdir(parents=True, exist_ok=True)

    def save_file(self, content: bytes, filename: str) -> str:
        """保存上传的 PDF 文件到 uploads/ 目录，用 UUID 重命名"""
        file_id = str(uuid.uuid4())  # 生成唯一文件名，防冲突
        saved_name = f"{file_id}{Path(filename).suffix or '.pdf'}"
        file_path = self.upload_dir / saved_name
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"已保存PDF文件: {file_path}")
        return str(file_path)

    def parse_paper(self, file_path: str, include_images: bool = True) -> dict:
        """解析论文：文本提取 + 图片提取 + 图表配对，返回结构化数据"""
        logger.info(f"开始解析论文: {file_path}")
        doc = fitz.open(file_path)  # 打开 PDF

        result = self._extract_text(doc)  # 提取文本、作者、公式、图表标题等
        result["full_text"] = self._clean_text(result["full_text"])  # 清理乱码字符

        # 提取图片并和图表标题配对
        result["images"] = self._extract_images(doc, file_path)

        # 把 Figure 和 Table 标题合并，统一去跟图片配对
        all_chart_items = []
        for f in result.get("figures", []):
            all_chart_items.append({**f, "type": "Figure"})
        for t in result.get("tables", []):
            all_chart_items.append({**t, "type": "Table"})
        result["figure_image_map"] = self._match_all_to_images(all_chart_items, result["images"], doc)

        doc.close()
        logger.info(f"解析完成: {result.get('title', '未知标题')}, {result.get('num_pages', 0)}页, "
                    f"图片: {len(result['images'])}张, 图表配对: {len(result['figure_image_map'])}个")
        return result

    def _extract_text(self, doc) -> dict:
        """从 PDF 提取文本和结构信息"""
        full_text = ""
        pages_text = []

        for i, page in enumerate(doc):
            text = page.get_text("text")  # 提取当前页纯文本
            pages_text.append(text)
            full_text += f"\n--- 第 {i + 1} 页 ---\n{text}"  # 拼接时加页码标记

        title = self._extract_title(pages_text[0] if pages_text else "", page0=doc[0] if len(doc) > 0 else None)
        authors = self._extract_authors(pages_text[0] if pages_text else "")
        formulas = self._extract_formulas(full_text)
        figures = self._extract_figures(doc)
        tables = self._extract_tables(doc)
        sections = self._extract_sections(full_text)
        num_pages = len(doc)

        return {
            "title": title, "authors": authors, "full_text": full_text,
            "pages_text": pages_text, "formulas": formulas, "figures": figures,
            "tables": tables, "sections": sections, "num_pages": num_pages
        }

    def _extract_title(self, first_page_text: str, page0=None) -> str:
        """从第一页提取论文标题（优先用大字体行，兜底黑名单过滤）"""
        # 版权类黑名单词，碰到直接跳过
        _blacklist = (
            "reproduce", "reprinted", "license", "creative commons",
            "downloaded", "personal use", "journalistic", "permission",
            "provided by", "copyright", "all rights reserved", "published in",
            "conference on", "proceedings of", "ieee", "acm", "springer",
            "arxiv",
        )

        lines = first_page_text.strip().split("\n")

        # 第一步：尝试用 PyMuPDF 找第一页最大字体的文本行
        if page0 is not None:
            try:
                blocks = page0.get_text("dict")["blocks"]
                font_candidates = []  # (字体大小, 文本, y坐标)
                for block in blocks:
                    if block.get("type") != 0:
                        continue
                    for line_info in block.get("lines", []):
                        text_parts = []
                        max_font = 0
                        for span in line_info.get("spans", []):
                            text_parts.append(span["text"])
                            fs = span.get("size", 0)
                            if fs > max_font:
                                max_font = fs
                        line_text = "".join(text_parts).strip()
                        if 8 < len(line_text) < 300:
                            font_candidates.append((max_font, line_text, line_info["bbox"][1]))
                if font_candidates:
                    # 按字体降序、y坐标升序（先出现的大字优先）
                    font_candidates.sort(key=lambda x: (-x[0], x[1]))
                    for fs, candidate_text, _ in font_candidates:
                        low = candidate_text.lower()
                        if not any(bw in low for bw in _blacklist):
                            return candidate_text
            except Exception:
                pass  # 字体提取失败，走兜底

        # 第二步：兜底 — 逐行过滤黑名单
        for line in lines[:10]:
            line = line.strip()
            low = line.lower()
            if (10 < len(line) < 200
                    and not any(c in line for c in ["@", "edu", "gmail", "com", "http", "arxiv"])
                    and not any(bw in low for bw in _blacklist)
                    and not low.startswith("abstract")):
                return line
        return "未识别标题"

    def _extract_authors(self, first_page_text: str) -> list:
        """用正则匹配"John Smith"格式的英文姓名"""
        authors = []
        for m in re.finditer(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", first_page_text):
            name = m.group(1).strip()
            if name not in ["University", "Institute", "Department", "School"]:  # 跳过机构名
                if 2 < len(name) < 50 and "@" not in name:
                    authors.append(name)
        return authors[:10]

    def _extract_formulas(self, text: str) -> list:
        """用正则匹配 $$...$$ 和 $...$ 提取 LaTeX 公式"""
        formulas = []
        # 显示公式 $$...$$ 或 \[...\]
        for m in re.finditer(r"\$\$([^$]+)\$\$|\\\[([^\]]+)\\\]", text):
            f = (m.group(1) or m.group(2)).strip()
            if len(f) > 2:
                formulas.append({"type": "display", "latex": f, "position": m.start()})
        # 行内公式 $...$
        for m in re.finditer(r"\$([^$]+)\$", text):
            f = m.group(1).strip()
            if len(f) > 2:
                formulas.append({"type": "inline", "latex": f, "position": m.start()})
        return formulas[:50]

    def _extract_figures(self, doc) -> list:
        """用 PyMuPDF 搜索 Figure 标题，获取位置坐标（同一页同编号只取首次匹配，防跨 block 重复）"""
        figures = []
        for page_num, page in enumerate(doc):
            blocks = page.get_text("blocks")  # 每个 block 包含 bbox + 文本
            for block in blocks:
                text = block[4] if len(block) > 4 else ""
                m = re.search(r"(?:Figure|Fig\.?)\s*([A-Z]?\d+[a-zA-Z]?)\s*[:\.]?\s*([^\n]+)", text, re.IGNORECASE)
                if m:  # 匹配到 "Figure 1: xxx" 或 "Fig. 3b xxx"
                    fig_num = m.group(1)
                    page = page_num + 1
                    # 同一页同编号只取首次匹配，防跨 block 重复
                    if any(f["number"] == fig_num and f["page"] == page for f in figures):
                        continue
                    figures.append({
                        "number": fig_num,
                        "caption": m.group(2).strip(),
                        "page": page,
                        "bbox": list(block[:4])  # (x0, y0, x1, y1) — 标题文字的位置
                    })
        return figures

    def _extract_tables(self, doc) -> list:
        """用 PyMuPDF 搜索 Table 标题，获取位置坐标（同一页同编号只取首次匹配，防跨 block 重复）"""
        tables = []
        for page_num, page in enumerate(doc):
            blocks = page.get_text("blocks")  # 每个 block 包含 bbox + 文本
            for block in blocks:
                text = block[4] if len(block) > 4 else ""
                m = re.search(r"(?:Table|Tab\.?)\s*([A-Z]?\d+[a-zA-Z]?)\s*[:\.]?\s*([^\n]+)", text, re.IGNORECASE)
                if m:  # 匹配到 "Table 1: xxx" 或 "Table A7 xxx"
                    tbl_num = m.group(1)
                    page = page_num + 1
                    # 同一页同编号只取首次匹配，防跨 block 重复
                    if any(t["number"] == tbl_num and t["page"] == page for t in tables):
                        continue
                    tables.append({
                        "number": tbl_num,
                        "caption": m.group(2).strip(),
                        "page": page,
                        "bbox": list(block[:4])  # (x0, y0, x1, y1) — 标题文字的位置
                    })
        return tables

    def _extract_sections(self, text: str) -> list:
        """匹配 "1. xxx" 或 "Abstract" 等章节标题"""
        sections = []
        for m in re.finditer(r"^(\d+\.?\s*[^\n]+)$|^(Abstract|Introduction|Related Work|Method|Experiment|Conclusion|References)[^\n]*$", text, re.MULTILINE | re.IGNORECASE):
            title = m.group(1) or m.group(2)
            if title and len(title) > 2:
                sections.append(title.strip())
        return sections

    def _extract_images(self, doc, file_path: str) -> List[Dict[str, Any]]:
        """抠出 PDF 内嵌图片存到 images/ 目录（按图片内容哈希去重）"""
        images = []
        seen_hashes = {}  # 哈希 → 文件名，相同图片只存一份

        def _dedup_and_save(image_bytes: bytes, ext: str, page_num: int, bbox, width: int, height: int):
            """检查图片哈希，重复则复用旧文件名，否则存新文件"""
            img_hash = hashlib.md5(image_bytes).hexdigest()
            if img_hash in seen_hashes:
                # 同一张图，复用之前的文件名和路径
                old_name = seen_hashes[img_hash]
                images.append({
                    "filename": old_name,
                    "path": str(self.image_output_dir / old_name),
                    "page": page_num,
                    "bbox": list(bbox) if bbox else None,
                    "width": width,
                    "height": height,
                    "format": ext.upper()
                })
                return
            # 新图，存到磁盘
            img_filename = f"img_{uuid.uuid4().hex[:8]}.{ext}"
            img_path = self.image_output_dir / img_filename
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            seen_hashes[img_hash] = img_filename
            images.append({
                "filename": img_filename,
                "path": str(img_path),
                "page": page_num,
                "bbox": list(bbox) if bbox else None,
                "width": width,
                "height": height,
                "format": ext.upper()
            })

        for page_num, page in enumerate(doc):
            extracted_count = 0  # 这一页成功提取了几张真图

            # 第一步：用 get_image_info 拿带坐标的图片
            try:
                # PyMuPDF库提供的方法，提取每一页的图片信息，包含坐标、宽度、高度等
                img_info_list = list(page.get_image_info())
            except Exception:
                img_info_list = []

            for img_info in img_info_list:
                # 提取图片信息
                xref = img_info.get("number", img_info.get("xref", 0))
                # 提取图片坐标
                bbox = img_info.get("bbox", None)
                if not xref:
                    continue
                try:
                    # 从文档中提取图片数据
                    base = doc.extract_image(xref)
                except Exception:
                    continue  # 假图（SMask/Form），跳过
                _dedup_and_save(
                    base["image"], base["ext"], page_num + 1,
                    bbox,
                    img_info.get("width", base.get("width", 0)),
                    img_info.get("height", base.get("height", 0))
                )
                extracted_count += 1

            # 第二步：get_image_info 一条真图没提出 → 用 get_images 兜底再扫一遍
            if extracted_count == 0:
                # 没有带坐标的图片，用 get_images 拿所有图片
                for img in page.get_images(full=True):
                    try:
                        # 从文档中提取图片数据
                        base = doc.extract_image(img[0])
                    except Exception:
                        continue  # 假图（SMask/Form），跳过
                    _dedup_and_save(
                        base["image"], base["ext"], page_num + 1,
                        None,
                        base.get("width", 0),
                        base.get("height", 0)
                    )
        return images

    def _match_all_to_images(self, items: list, images: list, doc) -> dict:
        """把 Figure/Table 标题和图片配对，没配对上的用页面截图兜底（矢量图表也能截到）"""
        result_map = {}  # 最终结果: {"Figure 1": "图片路径", "Table A7": "图片路径", "1": "图片路径", ...}
        if not items or not images:
            return result_map

        # 把图片按页码分组
        images_by_page = {}
        for img in images:
            page = img["page"]
            images_by_page.setdefault(page, []).append(img)

        # 把标题也按页码分组
        items_by_page = {}
        for item in items:
            page = item["page"]
            items_by_page.setdefault(page, []).append(item)

        # 逐页配对
        for page, page_imgs in images_by_page.items():
            page_items = items_by_page.get(page, [])
            if not page_items:
                continue

            # ---- 第一步：坐标精确匹配 ----
            # 标题在图表正下方，找标题正上方最近的图片
            matched = set()
            for item in page_items:
                item_bbox = item.get("bbox")
                if not item_bbox:
                    continue
                item_y0 = item_bbox[1]  # 标题上边缘 y 坐标
                best_img = None
                best_dist = float("inf")
                for img in page_imgs:
                    img_bbox = img.get("bbox")
                    if not img_bbox:
                        continue
                    img_y1 = img_bbox[3]  # 图片下边缘 y 坐标
                    if img_y1 > item_y0:  # 图片在标题下方，不可能对应
                        continue
                    dist = item_y0 - img_y1
                    if dist < best_dist:
                        best_dist = dist
                        best_img = img
                if best_img:
                    prefix = item.get("type", "Figure")  # Figure 或 Table
                    key_full = f"{prefix} {item['number']}"
                    result_map[key_full] = best_img["path"]
                    # 数字键优先保留 Figure 的，不被 Table 覆盖（Figures 先处理，先到先得）
                    if item["number"] not in result_map:
                        result_map[item["number"]] = best_img["path"]
                    matched.add(id(best_img))

            # ---- 第二步：顺序兜底 ----
            # 剩下没坐标的标题 + 没被第一步用掉的图片，按同页出现顺序一一对应
            unmatched_items = [it for it in page_items if f"{it.get('type','Figure')} {it['number']}" not in result_map]
            unmatched_imgs = [img for img in page_imgs if id(img) not in matched]

            for item, img in zip(unmatched_items, unmatched_imgs):
                prefix = item.get("type", "Figure")
                key_full = f"{prefix} {item['number']}"
                result_map[key_full] = img["path"]
                if item["number"] not in result_map:
                    result_map[item["number"]] = img["path"]

        # ---- 第三步：矢量图兜底 ----
        # 完全没配对上的（矢量图表），用页面截图裁剪提取
        for item in items:
            prefix = item.get("type", "Figure")
            key_full = f"{prefix} {item['number']}"
            if key_full not in result_map:
                cropped = self._crop_from_page(doc, item)
                if cropped:
                    result_map[key_full] = cropped
                    if item["number"] not in result_map:
                        result_map[item["number"]] = cropped

        # ---- 第四步：生成序号映射 ----
        # 按页码 → 页面内 Y 坐标排序，给每个图表排一个序号（用户说"第三张图"就能查到）
        sorted_items = sorted(items, key=lambda x: (
            x["page"],
            x.get("bbox", [0, 9999])[1] if x.get("bbox") else 9999
        ))
        order_list = []
        seen_keys = set()
        for item in sorted_items:
            prefix = item.get("type", "Figure")
            key_full = f"{prefix} {item['number']}"
            if key_full in result_map and key_full not in seen_keys:
                order_list.append(key_full)
                seen_keys.add(key_full)
        result_map["_order"] = order_list

        return result_map

    def _crop_from_page(self, doc, item: dict) -> Optional[str]:
        """把页面渲染成高清图片，按标题坐标裁剪出图表区域（矢量图也能截到）"""
        try:
            page_num = item["page"] - 1  # 页码从 0 开始
            page = doc[page_num]
            mat = fitz.Matrix(2, 2)  # 2 倍缩放 = ~144 DPI，清晰度够用

            bbox = item.get("bbox")
            if bbox:
                # 标题在图表下方，向上取 600pt 作为图表区域
                x0, y0, x1, y1 = bbox
                chart_top = max(0, y0 - 800)
                chart_left = max(0, x0 - 50)
                chart_right = min(page.rect.width, x1 + 200)
                chart_rect = fitz.Rect(chart_left, chart_top, chart_right, y0 + 20)
                pix = page.get_pixmap(matrix=mat, clip=chart_rect)
            else:
                # 没有坐标就截整页
                pix = page.get_pixmap(matrix=mat)

            img_filename = f"img_{uuid.uuid4().hex[:8]}.png"
            img_path = self.image_output_dir / img_filename
            pix.save(str(img_path))

            prefix = item.get("type", "Figure")
            logger.info(f"矢量图截图: {prefix} {item['number']} -> {img_filename}")
            return str(img_path)
        except Exception as e:
            logger.warning(f"截图裁剪失败 {item.get('type','Figure')} {item.get('number')}: {e}")
            return None

    def _clean_text(self, text: str) -> str:
        """去掉多余空行和控制字符"""
        text = re.sub(r"\n{3,}", "\n\n", text)  # 连续3个换行 → 2个换行
        text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", text)
        return text.strip()


# 单例
_pdf_parser: Optional[PDFParser] = None


def get_pdf_parser() -> PDFParser:
    """获取 PDF 解析器单例，全局复用"""
    global _pdf_parser
    if _pdf_parser is None:
        _pdf_parser = PDFParser()
    return _pdf_parser