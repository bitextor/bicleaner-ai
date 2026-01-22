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


class ServiceOverloadedError(Exception):
    """Raised when service is overloaded and cannot accept more requests."""
    pass


class BicleanerService:
    """Singleton service for bicleaner-ai scoring.

    Wraps the bicleaner-ai model with:
    - Singleton loading at startup (10-30 sec)
    - ThreadPoolExecutor for async inference (TensorFlow is blocking)
    - Semaphore-based backpressure to prevent overload
    - Inference timeout to prevent hung requests
    - TensorFlow GPU memory growth configuration
    """

    def __init__(self):
        self.model = None
        self.model_type = settings.model_type
        self.model_path = settings.model_path
        self.batch_size = settings.batch_size
        self.executor = ThreadPoolExecutor(max_workers=settings.inference_workers)
        self._loaded = False
        self._inference_semaphore: asyncio.Semaphore | None = None

    def _init_semaphore(self) -> None:
        """Initialize semaphore for backpressure control.

        Must be called from async context (after event loop is running).
        """
        if self._inference_semaphore is None:
            self._inference_semaphore = asyncio.Semaphore(
                settings.max_concurrent_requests
            )
            logger.info(
                f"Backpressure semaphore initialized: "
                f"max_concurrent_requests={settings.max_concurrent_requests}"
            )

    def _configure_gpu_memory(self) -> None:
        """Configure TensorFlow GPU memory growth.

        Prevents TF from grabbing all GPU memory at startup.
        Must be called BEFORE any TensorFlow imports.
        """
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(
                    f"TensorFlow GPU memory growth enabled for {len(gpus)} GPU(s)"
                )
            else:
                logger.info("No GPU detected, running on CPU")
        except Exception as e:
            logger.warning(f"Failed to configure GPU memory growth: {e}")

    def load(self) -> None:
        """Load bicleaner-ai model.

        Downloads from HuggingFace Hub if not cached locally.
        Configures GPU memory growth before loading.
        """
        if self._loaded:
            logger.info("Model already loaded, skipping")
            return

        logger.info(f"Loading model: {self.model_type} from {self.model_path}")
        logger.info(
            f"Performance settings: "
            f"inference_workers={settings.inference_workers}, "
            f"max_concurrent_requests={settings.max_concurrent_requests}, "
            f"inference_timeout={settings.inference_timeout_sec}s"
        )

        # Set TensorFlow environment before import
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Configure GPU memory growth BEFORE TensorFlow model loading
        self._configure_gpu_memory()

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
        logger.info(
            f"Model not found locally, downloading from HuggingFace: {model_path}"
        )
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

    @property
    def queue_depth(self) -> int:
        """Current number of requests waiting in queue."""
        if self._inference_semaphore is None:
            return 0
        # Semaphore value shows available slots, so occupied = max - available
        return settings.max_concurrent_requests - self._inference_semaphore._value

    async def score_batch_async(
        self, items: list[ScoreItem]
    ) -> list[ScoreResult]:
        """Score items asynchronously using ThreadPoolExecutor.

        Features:
        - Semaphore-based backpressure (returns 503 if overloaded)
        - Timeout protection (returns error if inference takes too long)

        Args:
            items: List of ScoreItem with id, source, target

        Returns:
            List of ScoreResult with id, success, score/error

        Raises:
            ServiceOverloadedError: If service is overloaded (semaphore timeout)
        """
        if not items:
            return []

        # Initialize semaphore on first async call
        self._init_semaphore()

        # Try to acquire semaphore with timeout (backpressure)
        try:
            await asyncio.wait_for(
                self._inference_semaphore.acquire(),
                timeout=settings.semaphore_acquire_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Service overloaded: queue_depth={self.queue_depth}, "
                f"rejecting request with {len(items)} items"
            )
            raise ServiceOverloadedError(
                f"Service overloaded, queue depth {self.queue_depth}. Try again later."
            )

        try:
            sources = [item.source for item in items]
            targets = [item.target for item in items]

            loop = asyncio.get_event_loop()

            try:
                # Run inference with timeout
                scores = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor, self.score_batch, sources, targets
                    ),
                    timeout=settings.inference_timeout_sec,
                )

                return [
                    ScoreResult(id=item.id, success=True, score=round(score, 3))
                    for item, score in zip(items, scores)
                ]

            except asyncio.TimeoutError:
                logger.error(
                    f"Inference timeout after {settings.inference_timeout_sec}s "
                    f"for batch of {len(items)} items"
                )
                return [
                    ScoreResult(
                        id=item.id,
                        success=False,
                        error=f"Inference timeout ({settings.inference_timeout_sec}s)",
                    )
                    for item in items
                ]

            except Exception as e:
                logger.exception("Scoring failed")
                return [
                    ScoreResult(id=item.id, success=False, error=str(e))
                    for item in items
                ]

        finally:
            # Always release semaphore
            self._inference_semaphore.release()

    def shutdown(self) -> None:
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
        logger.info("BicleanerService shutdown complete")
