import os
import sys
import unittest
import uuid


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import config
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.schemas import DocumentChunk, RetrievedChunk
from retrieval.sparse import SparseEncoderUnavailable, SparseVector, SpladeSparseEncoder
from storage import qdrant_store
from storage.qdrant_store import QdrantVectorStore


class FusionTests(unittest.TestCase):
    def test_rrf_merges_dense_and_sparse_results(self):
        dense = [
            RetrievedChunk(id="a", content="dense a", dense_score=0.9),
            RetrievedChunk(id="b", content="dense b", dense_score=0.8),
        ]
        sparse = [
            RetrievedChunk(id="b", content="sparse b", sparse_score=7.0),
            RetrievedChunk(id="c", content="sparse c", sparse_score=6.0),
        ]

        fused = reciprocal_rank_fusion(dense, sparse, k=60, limit=3)

        self.assertEqual(["b", "a", "c"], [chunk.id for chunk in fused])
        self.assertIsNotNone(fused[0].fusion_score)
        self.assertEqual(0.8, fused[0].dense_score)
        self.assertEqual(7.0, fused[0].sparse_score)


class SparseEncoderTests(unittest.TestCase):
    def test_sparse_encoder_requires_configured_model(self):
        original_model = config.QDRANT_SPARSE_MODEL
        try:
            config.QDRANT_SPARSE_MODEL = ""
            with self.assertRaises(SparseEncoderUnavailable):
                SpladeSparseEncoder().encode(["hello"])
        finally:
            config.QDRANT_SPARSE_MODEL = original_model


class QdrantPayloadTests(unittest.TestCase):
    def test_payload_rejects_old_source_only_metadata(self):
        original_model = config.QDRANT_SPARSE_MODEL
        try:
            config.QDRANT_SPARSE_MODEL = "test-splade"
            chunk = DocumentChunk(
                id="chunk-1",
                content="content",
                metadata={"source": "file.pdf"},
            )

            with self.assertRaises(qdrant_store.QdrantMetadataError):
                qdrant_store._build_payload(chunk)
        finally:
            config.QDRANT_SPARSE_MODEL = original_model


class QdrantStoreIntegrationTests(unittest.TestCase):
    def test_in_memory_qdrant_hybrid_search_enforces_user_filter(self):
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            self.skipTest("qdrant-client is not installed")

        original_collection = config.QDRANT_COLLECTION
        original_size = config.QDRANT_VECTOR_SIZE
        original_sparse_model = config.QDRANT_SPARSE_MODEL
        try:
            config.QDRANT_COLLECTION = f"test_{uuid.uuid4().hex}"
            config.QDRANT_VECTOR_SIZE = 2
            config.QDRANT_SPARSE_MODEL = "test-splade"

            store = QdrantVectorStore.__new__(QdrantVectorStore)
            store._client = QdrantClient(":memory:")

            user_one_chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                content="alpha content",
                metadata={
                    "user_id": "user-1",
                    "document_id": "doc-1",
                    "source": "alpha.pdf",
                    "file_type": "pdf",
                    "page_number": 1,
                },
            )
            user_two_chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                content="beta content",
                metadata={
                    "user_id": "user-2",
                    "document_id": "doc-2",
                    "source": "beta.pdf",
                    "file_type": "pdf",
                    "page_number": 1,
                },
            )

            store.upsert_chunks(
                [user_one_chunk, user_two_chunk],
                [[1.0, 0.0], [0.0, 1.0]],
                [
                    SparseVector(indices=[1], values=[1.0]),
                    SparseVector(indices=[2], values=[1.0]),
                ],
            )

            results = store.hybrid_search(
                [1.0, 0.0],
                SparseVector(indices=[1], values=[1.0]),
                user_id="user-1",
                top_k=5,
            )

            self.assertEqual(["alpha content"], [result.content for result in results])
            self.assertEqual({"user-1"}, {result.metadata["user_id"] for result in results})
        finally:
            config.QDRANT_COLLECTION = original_collection
            config.QDRANT_VECTOR_SIZE = original_size
            config.QDRANT_SPARSE_MODEL = original_sparse_model

    def test_payload_accepts_required_qdrant_metadata(self):
        original_model = config.QDRANT_SPARSE_MODEL
        try:
            config.QDRANT_SPARSE_MODEL = "test-splade"
            chunk = DocumentChunk(
                id="chunk-1",
                content="content",
                metadata={
                    "user_id": "user-1",
                    "document_id": "doc-1",
                    "source": "file.pdf",
                    "file_type": "pdf",
                    "page_number": 1,
                },
            )

            payload = qdrant_store._build_payload(chunk)

            self.assertEqual("user-1", payload["user_id"])
            self.assertEqual("chunk-1", payload["chunk_id"])
            self.assertEqual("content", payload["content"])
            self.assertEqual(config.EMBED_MODEL, payload["dense_model"])
            self.assertEqual("test-splade", payload["sparse_model"])
        finally:
            config.QDRANT_SPARSE_MODEL = original_model


if __name__ == "__main__":
    unittest.main()
