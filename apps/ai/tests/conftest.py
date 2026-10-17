import os

# env tối thiểu để app.main import được (fail-fast theo spec)
os.environ.setdefault("OLLAMA_URL", "http://ollama:11434")
os.environ.setdefault("OLLAMA_MODEL", "qwen3:4b")