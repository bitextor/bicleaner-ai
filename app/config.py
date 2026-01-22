"""Configuration for bicleaner-service."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="BICLEANER_")

    # Model settings
    model_type: str = "xlmr"
    model_path: str = "bitextor/bicleaner-ai-full-en-xx"

    # Inference settings
    batch_size: int = 32
    max_batch_size: int = 100
    inference_timeout_sec: int = 60

    # Performance settings
    inference_workers: int = 2  # ThreadPoolExecutor workers (2 for GPU, 4 for CPU)
    max_concurrent_requests: int = 10  # Semaphore limit for backpressure
    semaphore_acquire_timeout: float = 1.0  # Seconds to wait for semaphore slot

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8080


settings = Settings()
