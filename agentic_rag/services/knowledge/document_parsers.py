"""Document parsing helpers for the knowledge pipeline.

Parsing is independent of the pipeline's mutable index state, so it lives in
a mixin: ``KnowledgePipeline`` still exposes the same private hook names that
callers and tests rely on, while the implementation stays separate.
"""

import asyncio
from pathlib import Path

from agentic_rag.services.knowledge.content_list import ContentItem, ContentList
from agentic_rag.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentParsingMixin:
    """Turn files and raw input into ``ContentList`` objects."""

    def _text_to_content_list(self, text: str, source: str) -> ContentList:
        """Convert raw text to ContentList (with basic chunking)."""
        import re
        # Split on double newlines to create paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        items = []
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if para:
                items.append(ContentItem.from_text(para, page_idx=i // 5))
        return ContentList(items=items, source=str(source), source_type="text")

    def _dicts_to_content_list(self, dicts: list[dict], source: str) -> ContentList:
        """Convert list of dicts to ContentList (direct insertion API)."""
        items = []
        for d in dicts:
            ctype = ContentType(d.get("type", "text"))
            item = ContentItem(
                type=ctype,
                text=d.get("text", ""),
                page_idx=d.get("page_idx", 0),
                img_path=d.get("img_path"),
                img_url=d.get("img_url"),
                table_body=d.get("table_body"),
                table_caption=d.get("table_caption"),
                latex=d.get("latex"),
                video_path=d.get("video_path"),
                audio_path=d.get("audio_path"),
                metadata=d.get("metadata", {}),
            )
            items.append(item)
        return ContentList(items=items, source=str(source), source_type="content_list")

    async def _parse_file(self, source: str | Path, source_type: str) -> ContentList:
        """Parse a file into ContentList (placeholder for doc parsing integration)."""
        path = Path(source)
        if not path.exists():
            return self._text_to_content_list(str(source), str(source))

        suffix = path.suffix.lower()

        # Plain text / markdown
        if suffix in (".txt", ".md", ".markdown", ".py", ".json", ".yaml", ".yml", ".csv"):
            content = path.read_text()
            return self._text_to_content_list(content, str(source))

        # Image files
        if suffix in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"):
            return ContentList(
                items=[ContentItem.from_image(str(path))],
                source=str(source),
                source_type="image",
            )

        # Video files
        if suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            return ContentList(
                items=[ContentItem.from_video(str(path))],
                source=str(source),
                source_type="video",
            )

        # Audio files
        if suffix in (".mp3", ".wav", ".m4a", ".ogg", ".flac"):
            return ContentList(
                items=[ContentItem.from_audio(str(path))],
                source=str(source),
                source_type="audio",
            )

        # PDF — Docling is the default parser. OCR is opt-in via OCR__MODE.
        if suffix == ".pdf":
            from agentic_rag.config.settings import get_settings
            ocr_cfg = get_settings().ocr
            if ocr_cfg.enabled and getattr(ocr_cfg, "mode", "").strip():
                ocr_result = await self._parse_pdf_with_ocr(path)
                if ocr_result is not None and len(ocr_result.items) > 0:
                    return ocr_result
                logger.warning("OCR failed, falling back to Docling")
            else:
                logger.info("OCR mode not configured, using Docling")
            return await self._parse_with_docling(path)

        # Office documents — delegate to docling if available
        if suffix in (".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"):
            return await self._parse_with_docling(path)

        # Unknown — treat as text
        content = path.read_text()
        return self._text_to_content_list(content, str(source))

    async def _parse_pdf_with_ocr(self, path: Path) -> ContentList | None:
        """Parse a PDF using PaddleOCR-VL for OCR-based text extraction.

        Converts each page to an image, sends it to the OCR API, and returns
        a ContentList with per-page text. Returns None if OCR is unavailable
        or fails, so callers can fall back to other methods.
        """
        from agentic_rag.config.settings import get_settings
        ocr_cfg = get_settings().ocr
        if not ocr_cfg.enabled or not getattr(ocr_cfg, "mode", "").strip():
            return None
        if not ocr_cfg.model.strip():
            logger.warning("OCR__MODEL is empty, falling back to Docling")
            return None

        try:
            import fitz  # pymupdf
        except ImportError:
            logger.warning("pymupdf not installed, cannot render PDF pages")
            return None

        try:
            from openai import AsyncOpenAI
        except ImportError:
            logger.warning("openai not installed, cannot call OCR API")
            return None

        # Limit pages to avoid excessive API calls
        doc = fitz.open(str(path))
        total_pages = min(len(doc), ocr_cfg.max_pages)
        if total_pages <= 0:
            doc.close()
            return None

        logger.info("OCR parsing %s: %s pages via PaddleOCR-VL", path.name, total_pages)

        client = AsyncOpenAI(
            api_key=ocr_cfg.api_key,
            base_url=ocr_cfg.api_base,
            timeout=120.0,
        )

        items = []
        for page_idx in range(total_pages):
            try:
                page = doc[page_idx]
                # Render page to image at 200 DPI for good OCR quality
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("png")
                img_b64 = __import__("base64").b64encode(img_bytes).decode()
                data_uri = f"data:image/png;base64,{img_b64}"

                resp = await client.chat.completions.create(
                    model=ocr_cfg.model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {"type": "text", "text": "OCR:"},
                        ],
                    }],
                    temperature=0.0,
                    max_tokens=4096,
                )
                text = resp.choices[0].message.content or ""
                if text.strip():
                    items.append(ContentItem.from_text(
                        text.strip(),
                        page_idx=page_idx,
                        element_type="ocr_page",
                    ))
            except Exception as e:
                logger.warning("OCR page %s failed: %s", page_idx, e)
                if self._is_ocr_model_unavailable(e):
                    logger.warning(
                        "OCR model %r is unavailable, falling back to Docling",
                        ocr_cfg.model,
                    )
                    doc.close()
                    return None
                # Continue with other pages — don't abort the whole document

        doc.close()

        if not items:
            logger.warning("No text extracted from %s", path.name)
            return None

        logger.info("OCR extracted %s pages from %s", len(items), path.name)
        return ContentList(items=items, source=str(path), source_type="pdf")

    @staticmethod
    def _is_ocr_model_unavailable(error: Exception) -> bool:
        """Recognize OpenAI-compatible errors that mean the OCR model cannot run."""
        status_code = getattr(error, "status_code", None)
        code = str(getattr(error, "code", "") or "").lower()
        body = str(getattr(error, "body", "") or "").lower()
        message = f"{error} {body}".lower()
        markers = (
            "model_not_found",
            "model not found",
            "model does not exist",
            "unknown model",
            "no such model",
            "model unavailable",
            "model is not available",
        )
        return code == "model_not_found" or (
            status_code == 404 and "model" in message
        ) or any(marker in message for marker in markers)

    async def _parse_with_docling(self, path: Path) -> ContentList:
        """Parse PDF/Office documents with Docling."""
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = await asyncio.to_thread(converter.convert, str(path))
            doc = result.document

            items = []
            export_markdown = getattr(doc, "export_to_markdown", None)
            if callable(export_markdown):
                markdown = export_markdown().strip()
                if markdown:
                    items.append(ContentItem.from_text(
                        markdown,
                        page_idx=0,
                        element_type="docling_markdown",
                    ))
            else:
                # Compatibility with older Docling document objects.
                for i, element in enumerate(getattr(doc, "paragraphs", [])):
                    text = element.text if hasattr(element, "text") else str(element)
                    if text.strip():
                        items.append(ContentItem.from_text(
                            text.strip(),
                            page_idx=i // 10,
                            element_type="paragraph",
                        ))

            if items:
                logger.info("Docling parsed %s", path.name)
                return ContentList(
                    items=items,
                    source=str(path),
                    source_type=path.suffix.lstrip("."),
                )

        except ImportError:
            pass  # Docling not available
        except Exception as e:
            logger.exception("Docling parse error")

        # Fallback: try pymupdf for PDF
        if path.suffix.lower() == ".pdf":
            try:
                import fitz  # pymupdf
                items = []
                with fitz.open(str(path)) as pdf:
                    for page_idx, page in enumerate(pdf):
                        text = page.get_text().strip()
                        if text:
                            items.append(ContentItem.from_text(
                                text, page_idx=page_idx, element_type="pdf_page",
                            ))
                if items:
                    return ContentList(items=items, source=str(path), source_type="pdf")
            except ImportError:
                pass
            except Exception as e:
                logger.exception("pymupdf parse error")

        # Last resort: warn and return placeholder
        logger.warning(
            "Cannot parse %s: no parser available (install docling or pymupdf)",
            path.suffix,
        )
        return ContentList(
            items=[ContentItem.from_text(
                f"[Cannot parse {path.suffix} file: {path.name}. "
                f"Install 'docling' or 'pymupdf' for PDF/Office support.]"
            )],
            source=str(path),
            source_type=path.suffix,
        )
