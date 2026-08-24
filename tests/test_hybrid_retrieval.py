import tempfile
import unittest
from pathlib import Path

from agentic_rag.orchestration.l1_tools.rag_tools import RAGSearchTool
from agentic_rag.services.knowledge.content_list import ContentItem
from agentic_rag.services.knowledge.retrieval.full_text import FullTextIndex
from agentic_rag.services.knowledge.retrieval.hybrid import HybridRetriever, RetrievalResult
from agentic_rag.services.knowledge.pipeline import KnowledgePipeline


def _doc(chunk_id, text, score=0.0):
    return type("Doc", (), {
        "text": text,
        "score": score,
        "entity_id": chunk_id,
        "metadata": {
            "source": "doc-1",
            "content_type": "text",
            "entity_id": chunk_id,
        },
    })()


class _EmbeddingAdapter:
    async def embed_query(self, query):
        return [0.1, 0.2]


class _HybridStore:
    def __init__(self):
        self.vector_calls = 0
        self.bm25_calls = 0

    async def search(self, **kwargs):
        self.vector_calls += 1
        return [_doc("generic", "Download 0x02 回复超时", 0.9),
                _doc("faq", "USB 下载 INIT 报错 Wait input time out", 0.8)]

    async def full_text_search(self, **kwargs):
        self.bm25_calls += 1
        return [_doc("faq", "USB 下载 INIT 报错 Wait input time out"),
                _doc("solution", "USB 下载模式下请勿勾选 Uart Download")]


class _FullTextOnlyStore:
    def __init__(self):
        self.docs = []

    async def add_full_text(self, collection, docs):
        self.docs.extend(docs)


class _ContextPipeline:
    def __init__(self):
        self.requested_top_k = None

    async def retrieve(self, **kwargs):
        self.requested_top_k = kwargs["top_k"]
        return [
            RetrievalResult(
                ContentItem.from_text(f"chunk {index} " + "x" * 1000),
                score=1.0 - index * 0.1,
                entity_id=f"chunk-{index}",
                doc_id=f"doc-{index}",
            )
            for index in range(5)
        ]

    def list_files(self):
        return [
            {"doc_id": f"doc-{index}", "name": f"document-{index}.pdf"}
            for index in range(5)
        ]


class _UnrelatedYearPipeline(_ContextPipeline):
    async def retrieve(self, **kwargs):
        self.requested_top_k = kwargs["top_k"]
        return [
            RetrievalResult(
                ContentItem.from_text("AXDL 工具 2025 版本下载步骤和 USB 驱动安装说明。"),
                score=0.9,
                entity_id="axdl",
                doc_id="doc-0",
            )
        ]


class _SingleDocKBPipeline(_ContextPipeline):
    """A KB whose every chunk is both a BM25 and a vector hit (RRF 'bm25+vector').

    Reproduces the failure where an off-topic query against a single-document
    KB (e.g. a World Cup question against a drug-label index) was answered
    from that document because every chunk looked 'corroborated'.
    """

    async def retrieve(self, **kwargs):
        self.requested_top_k = kwargs["top_k"]
        chunks = [
            "布洛芬片说明书 【用法用量】口服。成人一次1片，若持续疼痛或发热，可间隔4～6小时重复用药1次，24小时不超过4次。",
            "【注意事项】本品为对症治疗药，不宜长期或大量使用，用于止痛不得超过5天。",
            "【有效期】24个月 【批准文号】国药准字H37020386",
        ]
        return [
            RetrievalResult(
                ContentItem.from_text(text),
                score=1.0 - index * 0.016,
                entity_id=f"chunk-{index}",
                doc_id="doc-0",
                source="bm25+vector",
            )
            for index, text in enumerate(chunks)
        ]

    def list_files(self):
        return [{"doc_id": "doc-0", "name": "布洛芬片.pdf"}]


class HybridRetrievalTests(unittest.IsolatedAsyncioTestCase):
    def test_fts5_bm25_finds_exact_mixed_language_faq(self):
        with tempfile.TemporaryDirectory() as directory:
            index = FullTextIndex(Path(directory) / "knowledge_fts.db")
            index.upsert([
                {
                    "id": "generic",
                    "text": "Download PC 等待镜像数据传输指令 0x02 的回复超时",
                    "source": "doc-1",
                    "content_type": "text",
                },
                {
                    "id": "faq",
                    "text": "USB 下载 INIT 报错 Wait input time out。USB 模式下请勿勾选 Uart Download。",
                    "source": "doc-1",
                    "content_type": "text",
                },
            ])

            results = index.search("USB 下载INIT报错Wait input time out原因", limit=5)

            self.assertEqual(results[0]["id"], "faq")

    async def test_hybrid_search_calls_dense_and_bm25_then_rrf_fuses(self):
        store = _HybridStore()
        retriever = HybridRetriever(
            vector_store=store,
            embedding_adapter=_EmbeddingAdapter(),
        )

        results = await retriever.retrieve(
            "USB 下载 INIT 报错 Wait input time out",
            mode="hybrid",
            top_k=3,
        )

        self.assertEqual(store.vector_calls, 1)
        self.assertEqual(store.bm25_calls, 1)
        self.assertEqual(results[0].entity_id, "faq")
        self.assertEqual(results[0].source, "bm25+vector")

    async def test_full_text_indexing_does_not_require_embedding(self):
        store = _FullTextOnlyStore()
        pipeline = KnowledgePipeline(vector_store=store, embedding_func=None)

        await pipeline.ingest(
            source="manual",
            content="USB 下载模式下请勿勾选 Uart Download。",
        )

        self.assertEqual(len(store.docs), 1)
        self.assertIn("Uart Download", store.docs[0]["text"])

    async def test_llm_context_is_limited_to_three_short_chunks(self):
        pipeline = _ContextPipeline()

        evidence = await RAGSearchTool(pipeline).execute("query", top_k=10)

        self.assertEqual(pipeline.requested_top_k, 12)
        self.assertIn("[R3]", evidence)
        self.assertNotIn("[R4]", evidence)
        content_block = evidence.split("===== SOURCES =====", 1)[0]
        self.assertLessEqual(len(content_block), 3 * (RAGSearchTool.MAX_CHUNK_CHARS + 120))

    async def test_year_query_rejects_unrelated_candidates(self):
        evidence = await RAGSearchTool(_UnrelatedYearPipeline()).execute(
            "2025中国出生人口数"
        )

        self.assertEqual(evidence, "No relevant content found in the knowledge base.")

    async def test_off_topic_query_rejects_single_doc_kb(self):
        tool = RAGSearchTool(_SingleDocKBPipeline())

        off_topic = await tool.execute("2026世界杯冠军队伍")
        self.assertEqual(off_topic, "No relevant content found in the knowledge base.")

        on_topic = await tool.execute("布洛芬每日用量")
        self.assertIn("[R1]", on_topic)
        self.assertIn("24小时不超过4次", on_topic)


if __name__ == "__main__":
    unittest.main()
