"""
GodNode v2026 – Transformer-Competitive Push
Author: venkat (@venkatandroid10), Hyderabad
Focus: Deeper composition, better fluency, larger scale

New in this version:
- Frozen multilingual encoder fallback (multilingual-e5-large or Indic variant)
- Lightweight sparse ANN index (numpy random-projection style, no faiss/annoy dep)
- Role-filler syntax templates (Subject/Verb/Object/Modifier)
- Trigram Markov + larger corpus loader
"""

import numpy as np
import re
import os
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional

try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("sentence-transformers not available → using random sparse fallback")

# ============================================================
# CONFIG
# ============================================================

DIM = 1024              # Smaller for real embeddings; scale up if GPU
SPARSITY = 0.05         # ~5% active dims
SEED = 42
ROLE_DIM_SPLIT = DIM // 4  # Reserve part for roles
CONTRADICTION_TAU = 0.70
CREATIVITY = 0.09
BEAM_WIDTH = 6
MAX_CORPUS_LINES = 50000  # safety cap

rng = np.random.default_rng(SEED)

# ============================================================
# VECTOR CORE – Sparse + Multilingual + Role-Filler
# ============================================================

class GodNodeCore:
    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.lexicon: Dict[str, np.ndarray] = {}
        self.role_vectors: Dict[str, np.ndarray] = {}
        self.encoder = None
        self._init_encoder()
        self.memory_index = []          # for ANN
        self.projection_matrix = rng.normal(size=(dim, dim // 32))  # random proj for ANN

    def _init_encoder(self):
        if HAS_TRANSFORMERS:
            try:
                # Best 2025–2026 multilingual / Indic option (change to Vyakyarth if local)
                self.encoder = SentenceTransformer('intfloat/multilingual-e5-large')
                print("Using multilingual-e5-large encoder")
            except:
                self.encoder = None
        if self.encoder is None:
            print("Fallback: sparse random bipolar vectors")

    def vec(self, text: str, normalize=True) -> np.ndarray:
        text = text.strip().lower()
        if text in self.lexicon:
            return self.lexicon[text].copy()

        if self.encoder:
            emb = self.encoder.encode(text, normalize_embeddings=True)
            v = emb.astype(np.float32)
            # Project to our dim + sign + sparsify
            if len(v) != self.dim:
                v = np.resize(v, self.dim)
            v = np.sign(v)
            # sparsify: keep top-k% magnitudes
            thresh = np.percentile(np.abs(v), 100 * (1 - SPARSITY))
            v[np.abs(v) < thresh] = 0
        else:
            # sparse random
            v = np.zeros(self.dim, dtype=np.float32)
            nnz = int(self.dim * SPARSITY)
            idx = rng.choice(self.dim, nnz, replace=False)
            v[idx] = rng.choice([-1.0, 1.0], nnz)
            v /= np.sqrt(nnz) + 1e-9

        if normalize:
            norm = np.linalg.norm(v) + 1e-9
            v /= norm
        self.lexicon[text] = v
        return v.copy()

    def role(self, role_name: str) -> np.ndarray:
        if role_name not in self.role_vectors:
            base = self.vec(role_name)
            # Put role in first ROLE_DIM_SPLIT dims, zero elsewhere
            r = np.zeros(self.dim)
            r[:ROLE_DIM_SPLIT] = base[:ROLE_DIM_SPLIT]
            self.role_vectors[role_name] = r
        return self.role_vectors[role_name].copy()

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    def bundle(self, vecs: List[np.ndarray]) -> np.ndarray:
        s = np.sum(vecs, axis=0)
        # re-sparsify
        thresh = np.percentile(np.abs(s), 100 * (1 - SPARSITY))
        s[np.abs(s) < thresh] = 0
        norm = np.linalg.norm(s) + 1e-9
        return s / norm

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def cleanup(self, query: np.ndarray, memory_vecs: List[np.ndarray], iters: int = 5):
        res = query.copy()
        for _ in range(iters):
            if np.linalg.norm(res) < 0.01:
                break
            sims = np.dot(memory_vecs, res)
            idx = np.argmax(sims)
            if sims[idx] < 0.2:
                break
            res -= sims[idx] * memory_vecs[idx]
        return res / (np.linalg.norm(res) + 1e-9)

    # Lightweight ANN: random projection + sorted buckets
    def build_index(self, items: List[Tuple[str, np.ndarray]]):
        self.memory_index = []
        for label, v in items:
            proj = np.dot(v, self.projection_matrix) > 0  # binary hash
            bucket_key = tuple(proj.astype(int).tolist())
            self.memory_index.append((bucket_key, label, v))

    def approx_nearest(self, q: np.ndarray, k: int = 8) -> List[Tuple[str, float]]:
        if not self.memory_index:
            return []
        q_proj = np.dot(q, self.projection_matrix) > 0
        q_key = tuple(q_proj.astype(int).tolist())
        candidates = [ (lbl, v) for key, lbl, v in self.memory_index if sum(key ^ q_key) <= 4 ]  # hamming <=4
        if len(candidates) < k:
            # fallback to more
            candidates = [ (lbl, v) for _, lbl, v in self.memory_index ]
        sims = [(lbl, self.cosine(q, v)) for lbl, v in candidates]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]

# ============================================================
# GODNODE MAIN
# ============================================================

class GodNode(GodNodeCore):
    def __init__(self, dim: int = DIM):
        super().__init__(dim)
        self.markov_trigram = defaultdict(lambda: defaultdict(int))  # (w1,w2) → w3
        self.vocab = []

    def train_on_corpus(self, corpus_paths: List[str]):
        all_text = ""
        for path in corpus_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    all_text += f.read() + "\n"
            else:
                print(f"Corpus file not found: {path}")

        tokens = [t for t in re.findall(r'\w+', all_text.lower()) if len(t) > 1]
        tokens = tokens[:MAX_CORPUS_LINES * 10]  # rough cap

        for i in range(len(tokens) - 2):
            ctx = (tokens[i], tokens[i+1])
            next_w = tokens[i+2]
            self.markov_trigram[ctx][next_w] += 1

        self.vocab = list(set(tokens))
        items = [(w, self.vec(w)) for w in self.vocab]
        self.build_index(items)
        print(f"Trained on {len(tokens)} tokens | Vocab: {len(self.vocab)}")

    def trigram_next(self, prev2: str, prev1: str):
        ctx = (prev2, prev1)
        if ctx in self.markov_trigram and self.markov_trigram[ctx]:
            ws, cs = zip(*self.markov_trigram[ctx].items())
            ps = np.array(cs) / sum(cs)
            return rng.choice(ws, p=ps)
        return None

    def compose(self, theme: str, length: int = 16):
        state = self.vec(theme)
        sequence = []
        prev2, prev1 = None, None

        for pos in range(length):
            # Role-filler: assign dynamic role
            role_name = ["Subject", "Verb", "Object", "Modifier"][pos % 4]
            role_v = self.role(role_name)

            context = self.bind(state, role_v)
            cleaned = self.cleanup(context, [v for _, v in self.memory_index or []])
            candidates = self.approx_nearest(cleaned, k=BEAM_WIDTH)

            scores = []
            for word, sim in candidates:
                markov_bonus = 0
                if prev2 and prev1:
                    trigram_w = self.trigram_next(prev2, prev1)
                    if trigram_w == word:
                        markov_bonus += 1.5
                diversity = -0.2 * sequence[-4:].count(word)
                total_score = sim + markov_bonus + diversity
                scores.append((word, total_score))

            if scores:
                scores.sort(key=lambda x: x[1], reverse=True)
                word = scores[0][0]
            else:
                word = theme

            sequence.append(word)
            prev2, prev1 = prev1, word

            wv = self.vec(word)
            state = self.bundle([state * 0.7, wv * 0.3])

        return " ".join(sequence)

# ============================================================
# DEMO & USAGE
# ============================================================

if __name__ == "__main__":
    print("GodNode 2026 – Enhanced with multilingual encoder + sparse index + role-filler syntax")
    node = GodNode()

    # Example: load larger corpus (replace with real paths!)
    # corpus_files = ["telugu_poetry.txt", "indic_corpus.txt", "english_poems.txt"]
    # For demo use small inline corpus
    sample_corpus = """
    జ్ఞానం నది వలె ప్రవహిస్తుంది knowledge flows like river
    విద్య దీపం భవిష్యత్తును వెలిగిస్తుంది education lights future
    శక్తి శాంతి కలిసి జీవితం అవుతుంది power peace together life
    """ * 20  # simulate larger

    with open("temp_large_corpus.txt", "w", encoding="utf-8") as f:
        f.write(sample_corpus)

    node.train_on_corpus(["temp_large_corpus.txt"])

    print("\nComposing longer poem around 'జ్ఞానం':")
    for i in range(4):
        poem = node.compose("జ్ఞానం", length=18)
        print(f"Poem {i+1}: {poem}\n")

    os.remove("temp_large_corpus.txt")  # cleanup
