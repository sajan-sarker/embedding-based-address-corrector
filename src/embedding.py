from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str, device: str | None = None, model_path: Path | None = None):
        self.model_name = model_name
        self.source = str(model_path) if model_path and model_path.is_dir() else model_name
        self.model = SentenceTransformer(self.source, device=device)
        self.device = str(self.model.device)
        self.dimension = self.model.get_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
