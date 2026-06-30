from dataclasses import dataclass

import config


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


class SparseEncoderUnavailable(RuntimeError):
    pass


class SpladeSparseEncoder:
    """Optional SPLADE sparse encoder.

    The project does not install heavy sparse-model dependencies by default. When
    `VECTOR_DB=qdrant` is selected, configure `QDRANT_SPARSE_MODEL` and install a
    compatible sparse encoder package before using hybrid retrieval.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = (model_name or config.QDRANT_SPARSE_MODEL).strip()
        self._model = None

    def encode(self, texts: list[str]) -> list[SparseVector]:
        if not self.model_name:
            raise SparseEncoderUnavailable(
                "QDRANT_SPARSE_MODEL is required for Qdrant SPLADE sparse vectors."
            )

        try:
            from fastembed import SparseTextEmbedding
        except ImportError as exc:
            raise SparseEncoderUnavailable(
                "fastembed is not installed. Install a compatible sparse encoder before "
                "using VECTOR_DB=qdrant hybrid retrieval."
            ) from exc

        if self._model is None:
            self._model = SparseTextEmbedding(model_name=self.model_name)

        encoded = []
        for sparse_embedding in self._model.embed(texts):
            indices = getattr(sparse_embedding, "indices", None)
            values = getattr(sparse_embedding, "values", None)
            if indices is None or values is None:
                raise SparseEncoderUnavailable(
                    "Sparse encoder returned an unsupported embedding shape."
                )
            encoded.append(
                SparseVector(
                    indices=[int(index) for index in indices],
                    values=[float(value) for value in values],
                )
            )
        return encoded
