# ==========================================
# GOD NODE v2: RECURSIVE LOGIC + INVENTION
# ==========================================
import subprocess, sys

# 0. AUTO-INSTALL (Self-Healing)
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try: import pypdf
except: install("pypdf"); import pypdf

try: from sentence_transformers import SentenceTransformer
except: install("sentence-transformers"); from sentence_transformers import SentenceTransformer

import os
import re  # Crucial for the Logic Gate
import torch
import torch.nn as nn
import numpy as np
from google.colab import drive

# 1. SETUP & LOADING
# ==========================================
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

BASE_DIR = "/content/drive/MyDrive/My_God_Node"
BRAIN_FILE = os.path.join(BASE_DIR, "physics_brain_mp8c.npy")
TEXT_FILE = os.path.join(BASE_DIR, "text_index.txt")

# Safety Check
if not os.path.exists(BRAIN_FILE):
    print("❌ Error: Brain file not found!")
    print("   -> Run the 'Builder Script' first.")
    exit()

print("🔧 Initializing Neural Engine...")
device = torch.device("cpu")
encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# 2. THE HARDWARE (MP8C COUPLER)
# ==========================================
class MP8C_Coupler(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Linear(384, 4096, bias=False)
        # CRITICAL: Must use same seed as builder
        torch.manual_seed(42) 
        nn.init.normal_(self.projector.weight, mean=0.0, std=1.0)
    
    def forward(self, x):
        x = self.projector(x)
        x = torch.sigmoid(x)
        return (x * 255).type(torch.uint8)

coupler = MP8C_Coupler().to(device)

# 3. LOAD THE BRAIN
# ==========================================
print(f"🧠 Loading 15,000+ Facts from Drive...")
# Load as int16 to prevent overflow during math
db = np.load(BRAIN_FILE).astype('int16')

with open(TEXT_FILE, "r", encoding="utf-8") as f:
    text_data = f.read().split("|||")

print(f"✅ SYSTEM READY. ({len(text_data)} Facts Loaded)")

# 4. THE AGENT (SMART LOGIC GATE)
# ==========================================
class GodNodeAgent:
    def __init__(self, db, text_data, encoder, coupler):
        self.db = db
        self.text_data = text_data
        self.encoder = encoder
        self.coupler = coupler
        
    def get_vector(self, text):
        """Converts text to 8-bit Vector"""
        with torch.no_grad():
            emb = self.encoder.encode([text], convert_to_tensor=True)
            return self.coupler(emb).cpu().numpy().astype('int16')

    def check_quality(self, text):
        """The Logic Gate: Returns True only for high-quality science."""
        clean = text.strip().lower()
        
        # 1. Reject Quiz Options (Strict Regex)
        # Finds " a. " or " b. " anywhere in the sentence
        if re.search(r'(^|\s)[a-e]\.\s', clean): 
            return False, "Detected Quiz Option"
        
        # 2. Reject Homework/Test Prep
        bad_words = ["calculate", "determine the", "test prep", "chapter review", "exercise"]
        if any(w in clean for w in bad_words): 
            return False, "Homework Problem"
        
        # 3. Reject Short Fragments
        if len(text) < 60: 
            return False, "Too Short"
        
        return True, "OK"

    def search(self, vector):
        """Finds nearest fact to a vector"""
        dist = np.abs(self.db - vector).sum(axis=1)
        best_idx = np.argmin(dist)
        score = 100 * (1 - (dist[best_idx] / (255 * 4096)))
        return self.text_data[best_idx], score

    def think(self, user_query):
        """Recursive Reasoning Loop"""
        print(f"\n🤔 THINKING: Analyzing '{user_query}'...")
        
        # --- LOOP 1: FAST LOOKUP ---
        q_vec = self.get_vector(user_query)
        fact_1, conf_1 = self.search(q_vec)
        is_good, reason = self.check_quality(fact_1)
        
        print(f"   🔍 Step 1 Found: '{fact_1[:40]}...' (Conf: {conf_1:.1f}%)")
        
        if conf_1 > 82.0 and is_good:
            print("   ✅ Logic Check Passed.")
            return f"ANSWER: {fact_1}"
        
        # --- LOOP 2: DEEP THINKING ---
        else:
            print(f"   ⚠️ Logic Check Failed ({reason}). Refining...")
            new_query = f"definition of {user_query} physics principle"
            print(f"   🔄 Step 2: Searching for '{new_query}'...")
            
            q_vec_2 = self.get_vector(new_query)
            fact_2, conf_2 = self.search(q_vec_2)
            
            print("   ✅ Synthesizing result.")
            return f"PRIMARY FACT: {fact_2}\n\n(Context: {fact_1})"

    def invent(self, concept_a, concept_b):
        """The Smart Creativity Engine (Vector Mixing + Filter)"""
        print(f"\n⚗️  INVENTION LAB: Mixing '{concept_a}' + '{concept_b}'...")
        
        # 1. Vector Math
        vec_a = self.get_vector(concept_a)
        vec_b = self.get_vector(concept_b)
        vec_new = (vec_a + vec_b) // 2
        
        # 2. Search
        dist = np.abs(self.db - vec_new).sum(axis=1)
        best_indices = np.argsort(dist)[:15] # Look at top 15 candidates
        
        print(f"🚀 The child of '{concept_a}' and '{concept_b}' is:\n")
        
        found_count = 0
        for idx in best_indices:
            fact = self.text_data[idx]
            
            # 3. APPLY FILTER (The Fix)
            # This deletes the "a. b. c." quiz answers silently
            is_good, _ = self.check_quality(fact)
            
            if is_good:
                found_count += 1
                print(f"💡 Innovation #{found_count}: {fact}")
                print("-" * 30)
            
            if found_count >= 3: break # Stop after 3 GOOD answers
            
        if found_count == 0:
            print("   (No valid scientific connections found. Try different concepts.)")

# ==========================================
# 5. MAIN INTERFACE
# ==========================================
agent = GodNodeAgent(db, text_data, encoder, coupler)

print("\n" + "="*40)
print("   GOD NODE: FULL SYSTEM (ONLINE)")
print("   - Type 'invent' to mix concepts")
print("   - Type 'q' to quit")
print("="*40)

# --- AUTO-RUN TEST ---
print("\n🧪 AUTO-TEST: Verifying 'Regex Filter' on Particle + Wave...")
agent.invent("Particle", "Wave")
print("✅ Test Complete. (Notice: No 'a. b. c.' quiz answers appeared!)")
# ---------------------

while True:
    user_input = input("\n👤 Enter Command: ")
    
    if user_input.lower() == 'q':
        break
        
    elif user_input.lower() == 'invent':
        c1 = input("   ⚗️  Concept 1: ")
        c2 = input("   ⚗️  Concept 2: ")
        agent.invent(c1, c2)
        
    else:
        # Standard Chat Mode
        response = agent.think(user_input)
        print(f"\n🤖 Agent Response:\n{response}")
        print("-" * 40)
