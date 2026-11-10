"""Download and save the embedding model (embedding.model in config.yaml) to embedding.model_path."""

import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import get_settings  # noqa: E402


def main() -> None:
    """Download the model from Hugging Face and save it locally."""
    settings = get_settings()
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {settings.model_name}...")
    model = SentenceTransformer(settings.model_name)
    model.save(str(settings.model_path))
    print(f"Model saved to {settings.model_path}")


if __name__ == "__main__":
    main()
