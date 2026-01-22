"""BicleanerService - wrapper for bicleaner-ai model."""

import asyncio
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

import yaml

from .config import settings
from .schemas import ScoreItem, ScoreResult

logger = logging.getLogger(__name__)


class BicleanerService:
    """Singleton service for bicleaner-ai scoring.

    Wraps the bicleaner-ai model with:
    - Singleton loading at startup (10-30 sec)
    - ThreadPoolExecutor for async inference (TensorFlow is blocking)
    - Batch scoring interface
    """

    def __init__(self):
        self.model = None
        self.model_type = settings.model_type
        self.model_path = settings.model_path
        self.batch_size = settings.batch_size
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._loaded = False

    def load(self) -> None:
        """Load bicleaner-ai model.

        Downloads from HuggingFace Hub if not cached locally.
        """
        if self._loaded:
            logger.info("Model already loaded, skipping")
            return

        logger.info(f"Loading model: {self.model_type} from {self.model_path}")

        # Set TensorFlow environment before import
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Import bicleaner_ai (triggers TensorFlow import)
        from bicleaner_ai.util import get_model

        # Resolve model path (download from HF if needed)
        metadata_path = self._resolve_model_path()

        # Load metadata.yaml
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)

        model_dir = os.path.dirname(metadata_path)
        classifier_settings = metadata["classifier_settings"]

        # Create and load model
        model_class = get_model(metadata["classifier_type"])
        self.model = model_class(model_dir, classifier_settings)
        self.model.load()

        self._loaded = True
        logger.info(f"Model loaded successfully: {metadata['classifier_type']}")

    def _resolve_model_path(self) -> str:
        """Resolve model path, downloading from HF if needed."""
        model_path = self.model_path

        # If path exists locally, use it
        if os.path.exists(model_path):
            if not re.match(r".*\.ya?ml$", model_path):
                model_path = os.path.join(model_path, "metadata.yaml")
            return model_path

        # Try downloading from HuggingFace Hub
        logger.info(f"Model not found locally, downloading from HuggingFace: {model_path}")
        from huggingface_hub import snapshot_download

        cache_path = snapshot_download(model_path)
        return os.path.join(cache_path, "metadata.yaml")

    def score_batch(self, sources: list[str], targets: list[str]) -> list[float]:
        """Score a batch of source-target pairs synchronously.

        Args:
            sources: List of source language texts
            targets: List of target language texts

        Returns:
            List of scores (0-1), higher = better quality
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        predictions = self.model.predict(
            sources, targets, batch_size=self.batch_size
        )
        # Flatten and convert to list
        return predictions.flatten().tolist()

    async def score_batch_async(
        self, items: list[ScoreItem]
    ) -> list[ScoreResult]:
        """Score items asynchronously using ThreadPoolExecutor.

        Args:
            items: List of ScoreItem with id, source, target

        Returns:
            List of ScoreResult with id, success, score/error
        """
        if not items:
            return []

        sources = [item.source for item in items]
        targets = [item.target for item in items]

        loop = asyncio.get_event_loop()

        try:
            scores = await loop.run_in_executor(
                self.executor, self.score_batch, sources, targets
            )

            return [
                ScoreResult(id=item.id, success=True, score=round(score, 3))
                for item, score in zip(items, scores)
            ]
        except Exception as e:
            logger.exception("Scoring failed")
            return [
                ScoreResult(id=item.id, success=False, error=str(e))
                for item in items
            ]

    def shutdown(self) -> None:
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
        logger.info("BicleanerService shutdown complete")
