"""
Download embedding weights once (Docker build / Railway custom build phase).

Keeps HF cache under HF_HOME so runtime can load with HF_HUB_OFFLINE=1
when the platform blocks huggingface.co.
Matches default in ai_modules/shared_models.py (all-MiniLM-L6-v2).
"""

from sentence_transformers import SentenceTransformer


def main() -> None:
    SentenceTransformer("all-MiniLM-L6-v2")


if __name__ == "__main__":
    main()
