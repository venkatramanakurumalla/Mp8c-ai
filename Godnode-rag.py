"""
GodNode-RAG: Large Context HDC with External Memory
Author: venkat (@venkatandroid10), Hyderabad

Key innovations:
- HDC for local coherence (working memory)
- FAISS/Annoy for episodic memory (unbounded storage)
- Hierarchical attention for RAG retrieval
- Write-back mechanism for memory consolidation
"""

import numpy as np
import faiss
import json
import hashlib
from collections import deque
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Callable
import re
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# ============================================================
# CONFIG
# ============================================================
DIM = 1024
SPARSITY = 0.05
MAX_WORKING_MEMORY = 8       # HDC chunks in active state
EPISODIC_CAPACITY = 1000000  # Unlimited via FAISS
RETRIEVAL_K = 16             # RAG candidates
CONSOLIDATION_THRESHOLD = 0.7 # When to write working→episodic

# ============================================================
# MEMORY HIERARCHY
# ============================================================

@dataclass
class MemoryTrace:
    """Unified memory entry"""
    id: str
    content: str                    # Raw text
    vector: np.ndarray              # HDC hypervector
    timestamp: float
    source: str                     # "working", "episodic", "external"
    metadata: Dict
    access_count: int = 0
    
    def to_dict(self):
        d = asdict(self)
        d['vector'] = self.vector.tobytes().hex()
        return d

class HDCWorkingMemory:
    """
    Active manipulation buffer (Miller's 4±1 chunks)
    Uses Kanerva-style hyperdimensional computing
    """
    def __init__(self, dim: int = DIM, capacity: int = MAX_WORKING_MEMORY):
        self.dim = dim
        self.capacity = capacity
        self.slots: List[MemoryTrace] = []
        self.composite_state = np.zeros(dim)
        self.temporal_trace = np.zeros(dim)  # Permutation chain
        
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b
    
    def bundle(self, vectors: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        if not vectors:
            return np.zeros(self.dim)
        if weights is None:
            weights = [1.0] * len(vectors)
        s = sum(w * v for w, v in zip(weights, vectors))
        # Prevent saturation: superposition cleanup
        norm = np.linalg.norm(s) + 1e-9
        return s / norm
    
    def permute(self, v: np.ndarray, shift: int) -> np.ndarray:
        return np.roll(v, shift)
    
    def encode_sequence(self, items: List[str], encoder: Callable[[str], np.ndarray]) -> np.ndarray:
        """Encode sequence via permutation (Plate, 2003)"""
        result = np.zeros(self.dim)
        for i, item in enumerate(items):
            vec = encoder(item)
            shifted = self.permute(vec, i)
            result += shifted
        return result / (np.linalg.norm(result) + 1e-9)
    
    def decode_sequence(self, composite: np.ndarray, decoder: Callable[[np.ndarray], str], 
                       max_len: int = 10) -> List[str]:
        """Reverse permutation to recover sequence"""
        result = []
        current = composite.copy()
        for i in range(max_len):
            # Unshift
            unshifted = self.permute(current, -i)
            # Decode
            item = decoder(unshifted)
            if item:
                result.append(item)
                # Subtract and continue
                item_vec = encoder(item)
                current -= self.permute(item_vec, i)
            else:
                break
        return result
    
    def add(self, trace: MemoryTrace):
        """Add to working memory with forgetting"""
        if len(self.slots) >= self.capacity:
            # Consolidate oldest to episodic (handled externally)
            self.slots.pop(0)
        
        self.slots.append(trace)
        
        # Update composite state with temporal encoding
        self.composite_state = self.bundle(
            [self.composite_state * 0.7, trace.vector * 0.3]
        )
        
        # Update temporal chain
        self.temporal_trace = self.bundle([
            self.permute(self.temporal_trace, 1),
            trace.vector
        ])
    
    def query_composite(self, query_vec: np.ndarray) -> List[Tuple[MemoryTrace, float]]:
        """Find relevant items in working memory"""
        scores = [(trace, np.dot(query_vec, trace.vector)) for trace in self.slots]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def get_context_window(self, n: int = 4) -> List[MemoryTrace]:
        """Get last n items (local context)"""
        return self.slots[-n:]

class EpisodicMemoryStore:
    """
    Unlimited storage via FAISS + metadata database
    Approximates hippocampal indexing
    """
    def __init__(self, dim: int = DIM):
        self.dim = dim
        # FAISS index for fast similarity search
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine if normalized)
        self.storage: Dict[str, MemoryTrace] = {}
        self.id_map: List[str] = []  # FAISS id → trace id
        
    def add(self, trace: MemoryTrace):
        """Store with vector indexing"""
        # Normalize for cosine similarity via inner product
        vec = trace.vector / (np.linalg.norm(trace.vector) + 1e-9)
        vec = vec.astype('float32').reshape(1, -1)
        
        self.index.add(vec)
        self.storage[trace.id] = trace
        self.id_map.append(trace.id)
        
    def retrieve(self, query_vec: np.ndarray, k: int = RETRIEVAL_K) -> List[MemoryTrace]:
        """RAG retrieval"""
        q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        q = q.astype('float32').reshape(1, -1)
        
        distances, indices = self.index.search(q, k)
        
        results = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.id_map):
                continue
            trace_id = self.id_map[idx]
            if trace_id in self.storage:
                trace = self.storage[trace_id]
                trace.access_count += 1
                results.append(trace)
        
        return results
    
    def retrieve_with_context(self, query_vec: np.ndarray, 
                             working_context: List[MemoryTrace],
                             k: int = RETRIEVAL_K) -> List[MemoryTrace]:
        """RAG enhanced by working memory context (hierarchical attention)"""
        # Boost query with working memory themes
        context_boost = np.mean([t.vector for t in working_context], axis=0) if working_context else 0
        enhanced_query = query_vec + 0.3 * context_boost
        enhanced_query /= np.linalg.norm(enhanced_query) + 1e-9
        
        return self.retrieve(enhanced_query, k)
    
    def consolidate_from_working(self, working_mem: HDCWorkingMemory, 
                                trigger: str = "capacity"):
        """Write working memory to long-term storage"""
        for trace in working_mem.slots:
            if trace.source == "working":
                trace.source = "episodic"
                self.add(trace)
        working_mem.slots = []
        working_mem.composite_state = np.zeros(self.dim)

class ExternalDocumentStore:
    """
    RAG over external documents (PDFs, books, web)
    Pre-chunked and indexed
    """
    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.chunks: List[Dict] = []  # {text, vector, source_doc}
        self.index = faiss.IndexFlatIP(dim)
        
    def index_documents(self, documents: List[str], encoder: Callable[[str], np.ndarray]):
        """Chunk and index documents"""
        for doc in documents:
            # Simple sentence chunking
            chunks = re.split(r'(?<=[.!?])\s+', doc)
            for chunk in chunks:
                if len(chunk) < 20:
                    continue
                vec = encoder(chunk)
                vec = vec / (np.linalg.norm(vec) + 1e-9)
                
                self.chunks.append({
                    'text': chunk,
                    'vector': vec,
                    'source': hashlib.md5(doc.encode()).hexdigest()[:8]
                })
                self.index.add(vec.astype('float32').reshape(1, -1))
    
    def query(self, query_vec: np.ndarray, k: int = 8) -> List[Dict]:
        """Standard RAG retrieval"""
        q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        q = q.astype('float32').reshape(1, -1)
        
        distances, indices = self.index.search(q, k)
        return [self.chunks[i] for i in indices[0] if 0 <= i < len(self.chunks)]

# ============================================================
# GODNODE-RAG INTEGRATION
# ============================================================

class GodNodeRAG:
    """
    Unified system: HDC working memory + Episodic store + External RAG
    """
    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.encoder = self._load_encoder()
        self.working = HDCWorkingMemory(dim)
        self.episodic = EpisodicMemoryStore(dim)
        self.external = ExternalDocumentStore(dim)
        
        # Generation state
        self.turn_count = 0
        self.session_trace = np.zeros(dim)  # Accumulated context
        
    def _load_encoder(self):
        if HAS_TRANSFORMERS:
            return SentenceTransformer('intfloat/multilingual-e5-large')
        return None
    
    def encode(self, text: str) -> np.ndarray:
        """Universal encoder with HDC sparsification"""
        if self.encoder:
            emb = self.encoder.encode(text, normalize_embeddings=True)
            v = emb.astype(np.float32)
            if len(v) != self.dim:
                v = np.resize(v, self.dim)
        else:
            v = np.random.randn(self.dim).astype(np.float32)
            
        # Bipolar + sparse
        v = np.sign(v)
        thresh = np.percentile(np.abs(v), 100 * (1 - SPARSITY))
        v[np.abs(v) < thresh] = 0
        return v / (np.linalg.norm(v) + 1e-9)
    
    def ingest(self, content: str, source: str = "session"):
        """Add content to working memory"""
        vec = self.encode(content)
        trace = MemoryTrace(
            id=hashlib.md5(f"{content}{datetime.now()}".encode()).hexdigest()[:16],
            content=content,
            vector=vec,
            timestamp=datetime.now().timestamp(),
            source=source,
            metadata={"turn": self.turn_count}
        )
        
        self.working.add(trace)
        self.session_trace = self.working.bundle([
            self.session_trace * 0.9, vec * 0.1
        ])
        
        # Consolidate if working memory full
        if len(self.working.slots) >= MAX_WORKING_MEMORY:
            self.episodic.consolidate_from_working(self.working)
    
    def rag_retrieve(self, query: str, use_external: bool = True) -> Dict[str, List]:
        """
        Hierarchical retrieval:
        1. Working memory (immediate context)
        2. Episodic memory (session history)
        3. External documents (knowledge base)
        """
        query_vec = self.encode(query)
        
        results = {
            'working': self.working.query_composite(query_vec),
            'episodic': self.episodic.retrieve_with_context(
                query_vec, self.working.slots
            ),
            'external': self.external.query(query_vec) if use_external else []
        }
        
        # Deduplicate and rank
        seen = set()
        merged = []
        for source in ['working', 'episodic', 'external']:
            for item in results[source]:
                if source == 'external':
                    content = item['text']
                    score = np.dot(query_vec, item['vector'])
                else:
                    content = item.content if hasattr(item, 'content') else item['content']
                    score = item[1] if isinstance(item, tuple) else np.dot(query_vec, item.vector)
                
                if content not in seen:
                    seen.add(content)
                    merged.append({
                        'content': content,
                        'score': float(score),
                        'source': source
                    })
        
        merged.sort(key=lambda x: x['score'], reverse=True)
        return {
            'query': query,
            'results': merged[:20],
            'working_context': [t.content for t in self.working.get_context_window()]
        }
    
    def generate_with_rag(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Generate text using RAG-augmented HDC
        """
        # Retrieve relevant context
        rag_results = self.rag_retrieve(prompt)
        
        # Build augmented prompt
        context_str = "\n".join([
            f"[{r['source']}] {r['content'][:100]}..." 
            for r in rag_results['results'][:5]
        ])
        
        augmented_prompt = f"""Context:
{context_str}

Working memory: {rag_results['working_context']}

Generate: {prompt}"""
        
        # HDC generation (simplified - would use full GodNode compose)
        # Here we simulate with template + RAG fusion
        return self._hdc_generate(augmented_prompt, rag_results)
    
    def _hdc_generate(self, prompt: str, rag_context: Dict) -> str:
        """Actual HDC generation with retrieved context"""
        # Encode all context elements
        context_vecs = [self.encode(r['content']) for r in rag_context['results'][:8]]
        prompt_vec = self.encode(prompt)
        
        # Create composite query: prompt ⊗ weighted_context
        if context_vecs:
            context_bundle = self.working.bundle(context_vecs, 
                                                weights=[1.0/(i+1) for i in range(len(context_vecs))])
            query = self.working.bind(prompt_vec, context_bundle)
        else:
            query = prompt_vec
        
        # Retrieve word candidates from episodic memory vocabulary
        # (In practice, use vocabulary index)
        candidates = self.episodic.retrieve(query, k=50)
        
        # Generate via HDC constrained walk
        words = []
        state = query.copy()
        
        for i in range(max_tokens // 5):  # Rough word estimate
            # Role-based generation (S-V-O-M cycle)
            role = ["Subject", "Verb", "Object", "Modifier"][i % 4]
            role_vec = self.working.encode_sequence([role], self.encode)
            
            # Bind state to role
            search_vec = self.working.bind(state, role_vec)
            
            # Find nearest in vocabulary (simplified)
            if candidates:
                best = max(candidates, key=lambda t: np.dot(search_vec, t.vector))
                word = best.content.split()[0] if ' ' in best.content else best.content
            else:
                word = "unknown"
            
            words.append(word)
            
            # Update state
            word_vec = self.encode(word)
            state = self.working.bundle([state * 0.7, word_vec * 0.3])
            state = self.working.permute(state, 1)  # Temporal shift
        
        return " ".join(words)
    
    def long_context_compose(self, theme: str, n_paragraphs: int = 5) -> str:
        """
        Generate long-form text with explicit memory management
        """
        paragraphs = []
        
        for p in range(n_paragraphs):
            # Retrieve relevant previous content
            if p > 0:
                query = self.encode(f"{theme} paragraph {p}")
                memories = self.episodic.retrieve_with_context(query, self.working.slots)
                context_hint = f" (continuing from: {memories[0].content[:50]}...)" if memories else ""
            else:
                context_hint = ""
            
            para_prompt = f"{theme}{context_hint}"
            para = self.generate_with_rag(para_prompt)
            paragraphs.append(para)
            
            # Ingest paragraph back into memory
            self.ingest(para, source=f"paragraph_{p}")
            
            # Periodic consolidation
            if p % 2 == 0:
                self.episodic.consolidate_from_working(self.working)
        
        return "\n\n".join(paragraphs)

# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    print("GodNode-RAG: Large Context HDC with External Memory")
    print("=" * 60)
    
    system = GodNodeRAG()
    
    # Ingest working documents
    documents = [
        "జ్ఞానం నది వలె ప్రవహిస్తుంది. Knowledge flows like a river.",
        "విద్య దీపం లాంటిది. Education is like a lamp.",
        "The Ganges carries wisdom from the Himalayas to the plains.",
        "Rivers connect mountains to oceans, just as knowledge connects generations.",
        "In Telugu tradition, water represents both purity and the flow of time."
    ]
    
    system.external.index_documents(documents, system.encode)
    
    # Simulate long conversation
    inputs = [
        "Tell me about knowledge",
        "How does it flow?",
        "What about education?",
        "Connect this to rivers",
        "Now write a poem combining these ideas"
    ]
    
    for inp in inputs:
        print(f"\nUser: {inp}")
        
        # Retrieve context
        rag = system.rag_retrieve(inp)
        print(f"Retrieved: {len(rag['results'])} items")
        print(f"Working memory: {rag['working_context']}")
        
        # Generate
        response = system.generate_with_rag(inp)
        print(f"Assistant: {response[:100]}...")
        
        # Store interaction
        system.ingest(f"User: {inp}\nAssistant: {response}", source="dialogue")
        system.turn_count += 1
    
    # Generate long-form
    print("\n" + "=" * 60)
    print("LONG-FORM COMPOSITION:")
    long_text = system.long_context_compose("knowledge and rivers", n_paragraphs=3)
    print(long_text)
