"""Env-driven config for the naive baseline. No SQLite, no admin UI."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Load a local .env file if present.

    This keeps the CLI usable when secrets are stored in .env instead of being
    exported into the current PowerShell session.
    """
    for path in (ROOT / ".env", ROOT.parent / ".env"):
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_dotenv()
DATA_DIR = Path(os.getenv("ACTIVIITY_DATA_DIR", str(ROOT / "data")))
PERSIST_DIR = Path(os.getenv("ACTIVIITY_BASELINE_PERSIST", str(ROOT / "baseline" / ".index")))


@dataclass(frozen=True)
class Cfg:
    # LLM
    llm_provider: str = os.getenv("LLM_PROVIDER", "openrouter")          # openrouter | ollama
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv("LLM_MODEL")
    ollama_base_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_llm_model: str = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:14b-instruct")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
     
    # Embeddings
    embed_provider: str = os.getenv("EMBED_PROVIDER", "openai")          # openai | ollama | sentence_transformers
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3:latest")
    st_embed_model: str = os.getenv("ST_EMBED_MODEL", "ibm-granite/granite-embedding-311m-multilingual-r2")
    st_embed_device: str = os.getenv("ST_EMBED_DEVICE", "cuda")
    
    hf_token: str = os.getenv("HF_TOKEN", "")

    # Retrieval
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    top_k: int = int(os.getenv("TOP_K", "5"))



    # Prompt (single system/QA prompt, French)
 #   qa_prompt_fr: str = (
  ##      "Contexte ci-dessous :\n---------------------\n{context_str}\n"
  #      "---------------------\nEn te fondant uniquement sur ce contexte "
  #      "(et non sur tes connaissances générales), réponds à la question "
  #      "en français, de façon structurée et concise. Si l'information n'est pas "
  #      "présente dans le con texte, dis explicitement : « Je ne trouve pas cette "
  #      "information dans la base Activiity. »\n"
  #      "Question : {query_str}\nRéponse :"
  #  )
  
    qa_prompt_fr: str = (
    "Contexte ci-dessous :\n---------------------\n{context_str}\n"
    "---------------------\nConsignes :\n"
    "1. Lis l'intégralité du contexte avant de décider.\n"
    "2. Si au moins un passage du contexte permet de répondre, même "
    "partiellement, tu DOIS répondre à partir de ce passage en français, "
    "de façon structurée et concise. Ne réponds jamais que l'information "
    "est absente si une partie de la réponse figure dans le contexte.\n"
    "3. Refuse uniquement si AUCUN passage ne concerne la question ; dans "
    "ce cas uniquement, réponds exactement : « Je ne trouve pas cette "
    "information dans la base Activiity. »\n"
    "Question : {query_str}\nRéponse :"
)

CFG = Cfg()
