import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agentic_rag.services.knowledge.content_list import ContentItem, ContentList
from agentic_rag.services.knowledge.pipeline import KnowledgePipeline


class DocumentParsingFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_ocr_model_falls_back_to_docling(self):
        pipeline = KnowledgePipeline()
        docling_result = ContentList(items=[ContentItem.from_text("parsed by docling")])
        pipeline._parse_with_docling = AsyncMock(return_value=docling_result)

        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "manual.pdf"
            pdf.write_bytes(b"%PDF-placeholder")
            settings = SimpleNamespace(
                ocr=SimpleNamespace(enabled=True, mode="ocr", model="", max_pages=50)
            )
            with patch("agentic_rag.config.settings.get_settings", return_value=settings):
                result = await pipeline._parse_file(pdf, "pdf")

        self.assertIs(result, docling_result)
        pipeline._parse_with_docling.assert_awaited_once_with(pdf)

    def test_detects_model_unavailable_errors(self):
        error = RuntimeError("The requested model does not exist")
        self.assertTrue(KnowledgePipeline._is_ocr_model_unavailable(error))
        self.assertFalse(KnowledgePipeline._is_ocr_model_unavailable(RuntimeError("timeout")))

    async def test_docling_is_default_when_ocr_mode_is_not_configured(self):
        pipeline = KnowledgePipeline()
        docling_result = ContentList(items=[ContentItem.from_text("docling default")])
        pipeline._parse_with_docling = AsyncMock(return_value=docling_result)

        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "manual.pdf"
            pdf.write_bytes(b"%PDF-placeholder")
            settings = SimpleNamespace(
                ocr=SimpleNamespace(enabled=True, mode="", model="unused", max_pages=50)
            )
            with patch("agentic_rag.config.settings.get_settings", return_value=settings):
                result = await pipeline._parse_file(pdf, "pdf")

        self.assertIs(result, docling_result)
        pipeline._parse_with_docling.assert_awaited_once_with(pdf)


if __name__ == "__main__":
    unittest.main()
