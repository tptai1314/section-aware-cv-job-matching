import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ollama_url: str
    ollama_model: str
    ollama_timeout_s: int
    max_upload_mb: int
    max_pages: int
    max_llm_chars: int
    ocr_enabled: bool
    ocr_langs: str
    strict_evidence: bool
    allow_legacy_doc: bool
    embed_model: str
    embed_device: str
    log_level: str
    chroma_path: str | None = None
    recommend_pool_size: int = 100
    recommend_top_k: int = 10
    internal_api_key: str | None = None
    # Hàng đợi xử lý file (docs/DESIGN_processing_queue.md §5)
    # worker_enabled mặc định False khi dựng Settings trực tiếp (test không muốn
    # thread nền); load_settings() đặt True theo env (prod bật).
    processing_db_path: str | None = None
    processing_files_dir: str | None = None
    processing_worker_enabled: bool = False
    job_max_attempts: int = 3
    job_stale_seconds: int = 300
    job_poll_interval_s: float = 1.0
    job_retry_base_delay_s: float = 1.0


REQUIRED_ENV = ("OLLAMA_URL", "OLLAMA_MODEL")


def _env_int(env: dict, name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(env: dict, name: str, default: bool) -> bool:
    raw = env.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_float(env: dict, name: str, default: float) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def load_settings(env: dict | None = None) -> Settings:
    env = env if env is not None else os.environ
    missing = sorted(name for name in REQUIRED_ENV if not env.get(name, "").strip())
    if missing:
        raise RuntimeError(f"missing required env vars: {', '.join(missing)}")
    return Settings(
        ollama_url=env["OLLAMA_URL"].strip().rstrip("/"),
        ollama_model=env["OLLAMA_MODEL"].strip(),
        ollama_timeout_s=_env_int(env, "OLLAMA_TIMEOUT_S", 180),
        max_upload_mb=_env_int(env, "MAX_UPLOAD_MB", 10),
        max_pages=_env_int(env, "MAX_PAGES", 20),
        max_llm_chars=_env_int(env, "MAX_LLM_CHARS", 24000),
        ocr_enabled=_env_bool(env, "OCR_ENABLED", False),
        ocr_langs=env.get("OCR_LANGS", "vie+eng").strip() or "vie+eng",
        strict_evidence=_env_bool(env, "STRICT_EVIDENCE", True),
        allow_legacy_doc=_env_bool(env, "ALLOW_LEGACY_DOC", False),
        embed_model=env.get("EMBED_MODEL", "BAAI/bge-m3").strip(),
        embed_device=env.get("EMBED_DEVICE", "cpu").strip() or "cpu",
        log_level=env.get("LOG_LEVEL", "INFO").strip() or "INFO",
        chroma_path=env.get("CHROMA_PATH", "").strip() or None,
        recommend_pool_size=_env_int(env, "RECOMMEND_POOL_SIZE", 100),
        recommend_top_k=_env_int(env, "RECOMMEND_TOP_K", 10),
        internal_api_key=env.get("INTERNAL_API_KEY", "").strip() or None,
        processing_db_path=env.get("PROCESSING_DB_PATH", "").strip() or None,
        processing_files_dir=env.get("PROCESSING_FILES_DIR", "").strip() or None,
        processing_worker_enabled=_env_bool(env, "PROCESSING_WORKER_ENABLED", True),
        job_max_attempts=max(1, _env_int(env, "JOB_MAX_ATTEMPTS", 3)),
        job_stale_seconds=max(1, _env_int(env, "JOB_STALE_SECONDS", 300)),
        job_poll_interval_s=max(0.01, _env_float(env, "JOB_POLL_INTERVAL_S", 1.0)),
        job_retry_base_delay_s=max(0.0, _env_float(env, "JOB_RETRY_BASE_DELAY_S", 1.0)),
    )