"""Load local config: .env secrets and YAML configs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]

_ENV_LOADED = False


def load_env(env_path: Optional[Path] = None) -> None:
    """Load a .env file at the repo root into os.environ (idempotent)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    path = env_path or (REPO_ROOT / ".env")
    if path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(path, override=False)
        except ImportError:
            _load_env_manual(path)
    _ENV_LOADED = True


def _load_env_manual(path: Path) -> None:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def get_database_url(required: bool = True) -> Optional[str]:
    """Return DATABASE_URL from the environment/.env, or raise."""
    load_env()
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url and required:
        raise RuntimeError(
            "DATABASE_URL is not set. Create a .env at the repo root with:\n"
            "  DATABASE_URL=postgresql://user:password@localhost:5432/dbname"
        )
    return url or None


def load_config(path: Path | str | None = None) -> dict:
    """Load the YAML config (default configs/base.yaml); {} when missing.

    Carries matching/retrieval/ranking settings and the section-aware
    ranking weights so they are not hardcoded (plan §13.5 / §7).
    """
    load_env()
    target = Path(path) if path is not None else REPO_ROOT / "configs" / "base.yaml"
    if not target.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - pyyaml is a requirement
        raise ImportError("pyyaml is required to load the config") from exc
    data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"config must be a mapping: {target}")
    return data


def get_ranking_weights(
    config: dict | None = None,
    default: dict[str, float] | None = None,
) -> dict[str, float]:
    """Weights for the hybrid ranking (plan §7.2), from `ranking.weights`."""
    config = config if config is not None else load_config()
    weights = (config.get("ranking") or {}).get("weights")
    if not weights:
        from ..ranking.hybrid import DEFAULT_WEIGHTS

        return dict(default or DEFAULT_WEIGHTS)
    return {str(k): float(v) for k, v in weights.items()}


def get_retrieval_settings(config: dict | None = None) -> dict:
    """Retrieval/matching settings with defaults (plan §13.4)."""
    config = config if config is not None else load_config()
    retrieval = config.get("retrieval") or {}
    matching = config.get("matching") or {}
    return {
        "pool_size": int(retrieval.get("pool_size", 100)),
        "top_k": int(retrieval.get("top_k", 10)),
        "status": str(retrieval.get("status", "READY")),
        "keyword_alpha": float(retrieval.get("keyword_alpha", 0.5)),
        "pairing": str(matching.get("pairing", "cross")),
        "aggregation": str(matching.get("aggregation", "mean")),
    }

