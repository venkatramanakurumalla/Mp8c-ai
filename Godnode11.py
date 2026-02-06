"""
GodNode v2026 – Fixed & Enhanced Version
Fixes:
- ANN filtering: better candidate selection + safe fallback
- cleanup(): uses correct memory vector list
- Encoder: safe dimension handling (pad/truncate + normalize)
"""

import numpy as np
import re
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("No sentence-transformers → using sparse random fallback")

# ============================================================
# CONFIG
# ============================================================

DIM = 1024
SPARSITY = 0.05
SEED = 42
ROLE_DIM_SPLIT = DIM // 4
CONTRADICTION_TAU = 0.70
CREATIVITY = 0.09
BEAM_WIDTH = 6
MAX_CORPUS_LINES = 50000

rng = np.random.default_rng(SEED)

# ============================================================
# CORE
# ============================================================

class GodNodeCore:
    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.lexicon: Dict[str, np.ndarray] = {}
        self.role_vectors: Dict[str, np.ndarray] = {}
        self.encoder = None
        self._init_encoder()
        # ANN support
        self.memory_vectors: List[np.ndarray] = []   # actual vectors for exact/cleanup
        self.memory_labels: List[str] = []           # parallel labels
        self.proj_matrix = rng.normal(size=(dim, dim // 32))  # for approx hashing

    def _init_encoder(self):
        if HAS_TRANSFORMERS:
            try:
                # Good multilingual model (2025–2026 relevant)
                self.encoder = SentenceTransformer('intfloat/multilingual-e5-large')
                print("Loaded multilingual-e5-large")
            except Exception as e:
                print(f"Encoder load failed: {e}")
               def permute(self, v: np.ndarray, steps: int = 1) -> np.ndarray:
        """The core shift operator: encodes 'Time' as 'Space'"""
        return np.roll(v, steps)

    def compose_with_narrative(self, theme: str, length: int = 16):
        state = self.vec(theme)
        # 1. Initialize a 'Narrative Vector' to store the trajectory
        narrative_state = np.zeros(self.dim)
        sequence = []
        prev2 = prev1 = None

        for pos in range(length):
            role_name = ["Subject", "Verb", "Object", "Modifier"][pos % 4]
            role_v = self.role(role_name)

            # 2. Bind the local theme with the global narrative history
            # This ensures the 'Subject' of line 5 knows the 'Object' of line 1
            context = self.bind(self.bundle([state, narrative_state]), role_v)
            
            cleaned = self.cleanup(context)
            candidates = self.approx_nearest(cleaned, k=BEAM_WIDTH)

            # [Candidate scoring logic remains the same]
            # ... (selecting 'word') ...

            sequence.append(word)
            wv = self.vec(word)

            # 3. PERMUTATION STEP: 
            # Shift the entire narrative to make room for the new thought
            # Narrative = ρ(Narrative) + Current_Word_Vector
            narrative_state = self.bundle([self.permute(narrative_state, 1), wv])
            
            # 4. Update local state
            state = self.bundle([state * 0.65, wv * 0.35])
            prev2, prev1 = prev1, word

        return " ".join(sequence)
     self.encoder = None
        def thinking_compose(self, theme: str, length: int = 16, variants: int = 3):
        """
        System 2 Reasoning: Generates variants, recalls them to check 
        coherence, and selects the most 'memorable' one.
        """
        best_poem = ""
        highest_coherence = -1.0

        for _ in range(variants):
            # 1. Generate a candidate via Narrative Permutation
            state = self.vec(theme)
            narrative_state = np.zeros(self.dim)
            candidate_seq = []
            
            for pos in range(length):
                role_v = self.role(["Subject", "Verb", "Object", "Modifier"][pos % 4])
                context = self.bind(self.bundle([state, narrative_state]), role_v)
                
                # Add a touch of creative noise per 'thought'
                context += rng.normal(0, 0.02, self.dim) 
                
                word = self.approx_nearest(self.cleanup(context), k=1)[0][0]
                candidate_seq.append(word)
                
                # Permute & Update
                wv = self.vec(word)
                narrative_state = self.bundle([self.permute(narrative_state, 1), wv])
                state = self.bundle([state * 0.6, wv * 0.4])

            # 2. RECALL STEP: Can the brain remember what it just thought?
            # We measure 'Coherence' by how many words are correctly recovered
            recalled_words = self.recall(narrative_state, length)
            
            # Coherence Score = Intersection / Total
            matches = sum(1 for a, b in zip(candidate_seq, recalled_words) if a == b)
            coherence = matches / length

            if coherence > highest_coherence:
                highest_coherence = coherence
                best_poem = " ".join(candidate_seq)

        return best_poem, highest_coherence

    def vec(self, text: str, normalize: bool = True) -> np.ndarray:
        text = text.strip().lower()
        if text in self.lexicon:
            return self.lexicon[text].copy()

        if self.encoder:
            emb = self.encoder.encode(text, normalize_embeddings=True)
            v = emb.astype(np.float32)
            # Safe resize: pad or truncate
            if len(v) < self.dim:
                pad = np.zeros(self.dim - len(v))
                v = np.concatenate([v, pad])
            elif len(v) > self.dim:
                v = v[:self.dim]
            v = np.sign(v)  # keep bipolar style
            # sparsify
            thresh = np.percentile(np.abs(v), 100 * (1 - SPARSITY))
            v[np.abs(v) < thresh] = 0
        else:
            # sparse random fallback
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
            r = np.zeros(self.dim)
            r[:ROLE_DIM_SPLIT] = base[:ROLE_DIM_SPLIT]
            self.role_vectors[role_name] = r
        return self.role_vectors[role_name].copy()

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    def bundle(self, vecs: List[np.ndarray]) -> np.ndarray:
        s = np.sum(vecs, axis=0)
        thresh = np.percentile(np.abs(s), 100 * (1 - SPARSITY))
        s[np.abs(s) < thresh] = 0
        norm = np.linalg.norm(s) + 1e-9
        return s / norm

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def cleanup(self, query: np.ndarray, iters: int = 5) -> np.ndarray:
        """Fixed: uses self.memory_vectors (correct list of actual vectors)"""
        if not self.memory_vectors:
            return query / (np.linalg.norm(query) + 1e-9)
        
        res = query.copy()
        for _ in range(iters):
            if np.linalg.norm(res) < 0.01:
                break
            sims = np.dot(self.memory_vectors, res)
            idx = np.argmax(sims)
            if sims[idx] < 0.20:
                break
            res -= sims[idx] * self.memory_vectors[idx]
            res /= (np.linalg.norm(res) + 1e-9)
        return res

    # Improved ANN: projection + distance-based candidate selection
    def build_index(self):
        self.memory_vectors = []
        self.memory_labels = []
        for text, vec in self.lexicon.items():
            self.memory_vectors.append(vec)
            self.memory_labels.append(text)
        print(f"Indexed {len(self.memory_vectors)} items")

    def approx_nearest(self, q: np.ndarray, k: int = 8) -> List[Tuple[str, float]]:
        if len(self.memory_vectors) <= 32:  # small → brute force
            sims = [(lbl, self.cosine(q, v)) for lbl, v in zip(self.memory_labels, self.memory_vectors)]
            sims.sort(key=lambda x: x[1], reverse=True)
            return sims[:k]

        # Approximate: project → find close in projection space
        q_proj = np.dot(q, self.proj_matrix) > 0
        candidates = []
        for i, v in enumerate(self.memory_vectors):
            v_proj = np.dot(v, self.proj_matrix) > 0
            hamming = np.sum(q_proj != v_proj)
            if hamming <= 6:  # loose threshold for candidates
                candidates.append((self.memory_labels[i], self.cosine(q, v)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]

# ============================================================
# MAIN GODNODE
# ============================================================

class GodNode(GodNodeCore):
    def __init__(self, dim: int = DIM):
        super().__init__(dim)
        self.markov_trigram = defaultdict(lambda: defaultdict(int))

    def train_on_corpus(self, corpus_paths: List[str]):
        all_text = ""
        for path in corpus_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    all_text += f.read() + "\n"

        tokens = [t for t in re.findall(r'\w+', all_text.lower()) if len(t) > 1]
        tokens = tokens[:MAX_CORPUS_LINES * 10]

        for i in range(len(tokens) - 2):
            ctx = (tokens[i], tokens[i+1])
            next_w = tokens[i+2]
            self.markov_trigram[ctx][next_w] += 1

        # Index after training
        self.build_index()

        print(f"Trained on ~{len(tokens)} tokens | Vocab size: {len(self.lexicon)}")

    def trigram_next(self, prev2: str, prev1: str) -> Optional[str]:
        ctx = (prev2, prev1)
        d = self.markov_trigram[ctx]
        if not d:
            return None
        ws, cs = zip(*d.items())
        ps = np.array(cs) / sum(cs)
        return rng.choice(ws, p=ps)

    def compose(self, theme: str, length: int = 16):
        state = self.vec(theme)
        sequence = []
        prev2 = prev1 = None

        for pos in range(length):
            role_name = ["Subject", "Verb", "Object", "Modifier"][pos % 4]
            role_v = self.role(role_name)
            context = self.bind(state, role_v)
            cleaned = self.cleanup(context)   # now correct memory
            candidates = self.approx_nearest(cleaned, k=BEAM_WIDTH)

            scores = []
            for word, sim in candidates:
                markov_bonus = 0
                if prev2 and prev1:
                    pred = self.trigram_next(prev2, prev1)
                    if pred == word:
                        markov_bonus += 1.6
                diversity = -0.25 * sequence[-5:].count(word)
                total = sim + markov_bonus + diversity
                scores.append((word, total))

            if scores:
                scores.sort(key=lambda x: x[1], reverse=True)
                word = scores[0][0]
            else:
                word = theme

            sequence.append(word)
            prev2, prev1 = prev1, word

            wv = self.vec(word)
            state = self.bundle([state * 0.68, wv * 0.32])

        return " ".join(sequence)

# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    print("GodNode – Fixed ANN, cleanup & encoder resize")
    node = GodNode()

    # Small inline corpus (replace with real large file paths)
    sample_text = """
    జ్ఞానం నది వలె ప్రవహిస్తుంది knowledge flows like river
    విద్య దీపం భవిష్యత్తును వెలిగిస్తుంది education lights future
    శక్తి శాంతి కలిసి జీవితం అవుతుంది power peace together life
    """ * 50

    with open("temp_corpus.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)

    node.train_on_corpus(["temp_corpus.txt"])

    print("\nGenerated poems:")
    for i in range(3):
        print(f"\nPoem {i+1}:")
        print(node.compose("జ్ఞానం", length=18))

    os.remove("temp_corpus.txt")
