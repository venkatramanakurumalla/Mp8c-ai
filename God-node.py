# ==============================================================================
# GOD NODE v3: NEURO-SYMBOLIC HYPERDIMENSIONAL ARCHITECTURE
# Author: Venkataramana Kurumalla
# Papers Implemented: MPxC, Pocket University, Shai
# ==============================================================================

import os
import sys
import re
import torch
import torch.nn as nn
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for PDF reading

# ==============================================================================
# 1. THE PHYSICS CORE (MPxC / MP8C) [Ref: E_pdf.pdf]
# ==============================================================================
# [cite_start]"A deterministic and quantized semantic coupling framework" [cite: 7]
# [cite_start]"Frozen transformer embeddings, fixed random projections" [cite: 8]

class MP8C_Coupler(nn.Module):
    def __init__(self, input_dim=384, hdc_dim=4096):
        super().__init__()
        self.projector = nn.Linear(input_dim, hdc_dim, bias=False)
        
        # DETERMINISM ENFORCER [Ref: shai.pdf, Section 5]
        # [cite_start]"Initialized with a fixed random seed to ensure determinism" [cite: 201]
        torch.manual_seed(42) 
        nn.init.normal_(self.projector.weight, mean=0.0, std=1.0)
    
    def forward(self, x):
        # [cite_start]1. Projection: z = We [cite: 42]
        x = self.projector(x)
        # [cite_start]2. Activation: Sigmoid [cite: 44]
        x = torch.sigmoid(x)
        # [cite_start]3. Quantization: 8-bit Integer (0-255) [cite: 46]
        return (x * 255).type(torch.uint8)

# ==============================================================================
# 2. THE BUILDER (Ingestor) [Ref: pocket.pdf]
# ==============================================================================
# [cite_start]"Compressing 15,000+ physics vectors into a 60MB index" [cite: 114]

class KnowledgeBuilder:
    def __init__(self, output_name="god_node_brain"):
        self.output_name = output_name
        self.device = torch.device("cpu") # Offline/Edge safe
        print("🔧 Initializing MP8C Encoder...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.coupler = MP8C_Coupler().to(self.device)
        self.coupler.eval()

    def clean_text(self, text):
        """Sanitizes text from PDF artifacts"""
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text) # Fix hyphen-splits
        text = text.replace('\n', ' ')
        return re.sub(r'\s+', ' ', text).strip()

    def ingest_folder(self, folder_path):
        text_data = []
        vector_data = []
        
        print(f"📂 Scanning {folder_path}...")
        files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        
        for file in files:
            path = os.path.join(folder_path, file)
            print(f"   -> Reading: {file}")
            try:
                doc = fitz.open(path)
                full_text = "".join([page.get_text() for page in doc])
                clean = self.clean_text(full_text)
                
                # Chunking (Window of 3 sentences)
                sentences = re.split(r'(?<=[.!?]) +', clean)
                chunks = [" ".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
                
                # Filter short noise
                valid_chunks = [c for c in chunks if len(c) > 50]
                text_data.extend(valid_chunks)
                
                # Batch Encode
                if valid_chunks:
                    with torch.no_grad():
                        emb = self.encoder.encode(valid_chunks, convert_to_tensor=True)
                        quantized = self.coupler(emb)
                        vector_data.append(quantized.cpu().numpy())
                        
            except Exception as e:
                print(f"   ⚠️ Error: {e}")

        if not vector_data:
            print("❌ No data found.")
            return

        # Save Brain
        print("💾 Saving 8-bit Quantized Brain...")
        final_vectors = np.vstack(vector_data)
        np.save(f"{self.output_name}.npy", final_vectors)
        
        with open(f"{self.output_name}_text.pkl", "wb") as f:
            pickle.dump(text_data, f)
            
        print(f"✅ Build Complete. Brain Size: {final_vectors.nbytes / (1024**2):.2f} MB")

# ==============================================================================
# 3. THE AGENT (The Soul) [Ref: shai.pdf, pocket.pdf]
# ==============================================================================
# [cite_start]"Recursive logic... Concept Synthesis... Reliable reasoning" [cite: 169]

class GodNodeSoul:
    def __init__(self, brain_path="god_node_brain"):
        print("🧠 Waking up the God Node...")
        self.device = torch.device("cpu")
        
        # Load Hardware
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.coupler = MP8C_Coupler().to(self.device)
        
        # Load Memory
        try:
            self.memory = np.load(f"{brain_path}.npy").astype('int16') # int16 for math safety
            with open(f"{brain_path}_text.pkl", "rb") as f:
                self.text_db = pickle.load(f)
            print(f"   -> Loaded {len(self.text_db)} concepts.")
        except:
            print("❌ Brain not found! Run 'Builder' mode first.")
            sys.exit()

        # Pre-load Emotion Anchors (The "Soul" Vector Update)
        print("❤️  Loading Emotional Cortex...")
        self.v_happy = self._encode_internal("joy hope success light clear")
        self.v_serious = self._encode_internal("warning danger critical caution")
        self.v_curious = self._encode_internal("mystery wonder why how universe")

    def _encode_internal(self, text):
        with torch.no_grad():
            emb = self.encoder.encode([text], convert_to_tensor=True)
            return self.coupler(emb).cpu().numpy().flatten().astype('int16')

    # --- A. TITANIUM FILTER (Guardrail) [Ref: pocket.pdf] ---
    def titanium_filter(self, text):
        """
        [cite_start]'Rejects if text contains ambiguity indicators... output: deterministic definitions' [cite: 129]
        """
        # 1. Reject Quiz Options
        if re.search(r'(^|\s)[a-d]\.\s', text): return False
        # 2. Reject Homework Commands
        if any(x in text.lower() for x in ['calculate', 'find the value', 'exercise']): return False
        # 3. Reject Short Noise
        if len(text) < 40: return False
        return True

    # --- B. CORE SEARCH (L1 Distance) [Ref: E_pdf.pdf] ---
    def search(self, vector, top_k=1):
        # [cite_start]"D(q1, q2) = Sum |q1 - q2|" [cite: 53]
        dists = np.sum(np.abs(self.memory - vector), axis=1)
        indices = np.argsort(dists)[:top_k*3] # Fetch extra for filtering
        
        results = []
        for idx in indices:
            candidate = self.text_db[idx]
            if self.titanium_filter(candidate):
                results.append(candidate)
                if len(results) >= top_k: break
        
        return results if results else ["No valid data found."]

    # --- C. REASONING & EMOTION (The "Soul" Update) ---
    def think(self, query, mood="neutral"):
        """
        [cite_start]Combines Recursive Logic [cite: 213] with Emotional Steering.
        """
        print(f"\n🤔 Thinking about: '{query}' (Mood: {mood})")
        
        # 1. Vectorize Query
        q_vec = self._encode_internal(query)
        
        # 2. Apply Emotional Steering (Vector Addition)
        if mood == "happy": q_vec = q_vec + (self.v_happy // 4)
        if mood == "serious": q_vec = q_vec + (self.v_serious // 4)
        if mood == "curious": q_vec = q_vec + (self.v_curious // 4)
        
        # 3. Recursive Hop (Multi-step Reasoning)
        # Hop 1: Get definition
        fact_1 = self.search(q_vec, top_k=1)[0]
        
        # Hop 2: Contextualize (Query + Fact 1) // 2
        # [cite_start]"Concept Synthesis... via Vector Algebra" [cite: 66, 222]
        v_fact1 = self._encode_internal(fact_1)
        v_hop = (q_vec + v_fact1) // 2 
        fact_2 = self.search(v_hop, top_k=1)[0]
        
        return f"📝 FACT: {fact_1}\n💡 INSIGHT: {fact_2}"

    # --- D. LOGICAL ANALOGY (A is to B as C is to ?) ---
    def analogy(self, a, b, c):
        """
        Solves: A -> B :: C -> ?
        Math: Target = B - A + C
        """
        print(f"\n⚗️  Analogy: '{a}' is to '{b}' as '{c}' is to...?")
        va = self._encode_internal(a)
        vb = self._encode_internal(b)
        vc = self._encode_internal(c)
        
        # Symbolic Logic in Vector Space
        target = (vb - va) + vc
        # Clip to ensure valid 8-bit range logic (0-255)
        target = np.clip(target, 0, 255)
        
        result = self.search(target, top_k=1)[0]
        return f"🔍 RESULT: {result}"

    # --- E. INVENTION (Creative Synthesis) ---
    def invent(self, concept_a, concept_b):
        """
        [cite_start]"q(A+B) = floor((qA + qB) / 2)" [cite: 66]
        """
        print(f"\n🚀 Inventing: {concept_a} + {concept_b}")
        va = self._encode_internal(concept_a)
        vb = self._encode_internal(concept_b)
        
        # Creative Noise (Temperature)
        noise = np.random.randint(-5, 5, size=4096).astype('int16')
        v_mix = ((va + vb) // 2) + noise
        
        result = self.search(v_mix, top_k=1)[0]
        return f"✨ CONCEPT: {result}"

# ==============================================================================
# 4. MAIN INTERFACE
# ==============================================================================

if __name__ == "__main__":
    print("=========================================")
    print("   GOD NODE v3: NEURO-SYMBOLIC CORE      ")
    print("=========================================")
    mode = input("Select Mode: [1] Build Brain  [2] Activate Soul: ")
    
    if mode == "1":
        folder = input("Enter path to PDF folder: ")
        builder = KnowledgeBuilder()
        builder.ingest_folder(folder)
        
    elif mode == "2":
        agent = GodNodeSoul()
        print("\nCommands: 'invent', 'analogy', 'quit', or just type a question.")
        print("Optional: Type 'happy:', 'serious:', or 'curious:' before query.")
        
        while True:
            q = input("\n👤 YOU: ").strip()
            if q == 'quit': break
            
            if q.startswith('invent '):
                parts = q.replace('invent ', '').split('+')
                if len(parts) == 2: print(agent.invent(parts[0], parts[1]))
            
            elif q.startswith('analogy '):
                # Format: analogy A, B, C
                parts = q.replace('analogy ', '').split(',')
                if len(parts) == 3: print(agent.analogy(parts[0], parts[1], parts[2]))
                
            else:
                # Check for mood prefix
                mood = "neutral"
                if "happy:" in q: mood="happy"; q=q.replace("happy:","")
                if "serious:" in q: mood="serious"; q=q.replace("serious:","")
                if "curious:" in q: mood="curious"; q=q.replace("curious:","")
                
                print(agent.think(q, mood=mood))
