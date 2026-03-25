import hashlib
import logging
import math
import re
from typing import List

logger = logging.getLogger(__name__)


class TextEmbedder:
    """Simple, deterministic text embedder that produces 256-dim vectors.

    This implementation intentionally avoids heavyweight ML models and
    external APIs. It uses a feature-hashing style approach to map tokens
    into a fixed-size vector space and L2-normalizes the result.

    While this is not a state-of-the-art semantic model, it is completely
    self-contained and sufficient for demonstrating correct vector index
    configuration and search wiring.
    """

    TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("Embedding dimension must be positive")
        self.dim = dim
        logger.info("Initialized TextEmbedder with dim=%d", dim)

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in self.TOKEN_PATTERN.findall(text.lower()) if t]

    def embed(self, text: str) -> List[float]:
        """Embed text into a fixed-length dense vector.

        Algorithm:
        - Tokenize on word characters.
        - For each token, compute an MD5 hash.
        - Use the hash to choose an index in [0, dim) and a sign (+1 / -1).
        - Accumulate counts and L2-normalize at the end.
        """

        tokens = self._tokenize(text or "")
        if not tokens:
            # Return a zero vector with a 1 in the first dimension to
            # avoid division by zero and keep deterministic behavior.
            vec = [0.0] * self.dim
            vec[0] = 1.0
            return vec

        vec = [0.0] * self.dim

        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            index = h % self.dim
            sign = 1.0 if ((h >> 1) & 1) else -1.0
            vec[index] += sign

        norm_sq = sum(v * v for v in vec)
        if norm_sq == 0.0:
            # Extremely unlikely, but keep behavior well-defined.
            vec[0] = 1.0
            return vec

        norm = math.sqrt(norm_sq)
        return [v / norm for v in vec]
