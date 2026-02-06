"""
GodNode-Story: Long-form Narrative Generation via Hierarchical HDC
Author: venkat (@venkatandroid10), Hyderabad

Core innovation: Narrative superstructure with character/setting/plot hypervectors
"""

import numpy as np
import faiss
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import re
from enum import Enum, auto

try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# ============================================================
# NARRATIVE CONFIGURATION
# ============================================================

DIM = 2048  # Larger for complex narrative binding
SPARSITY = 0.04
MAX_SCENE_LENGTH = 500      # words
MAX_CHAPTER_LENGTH = 5      # scenes
PLOT_POINTS_PER_ACT = 3

class StoryPhase(Enum):
    EXPOSITION = auto()
    RISING_ACTION = auto()
    CLIMAX = auto()
    FALLING_ACTION = auto()
    RESOLUTION = auto()

# ============================================================
# NARRATIVE ENTITIES (Characters, Settings, Events)
# ============================================================

@dataclass
class Character:
    name: str
    traits: Dict[str, np.ndarray]  # "brave", "angry" → vectors
    relationships: Dict[str, float]  # character name → bond strength
    arc_vector: np.ndarray  # How they change
    current_state: np.ndarray  # Moment-to-moment
    
    def __post_init__(self):
        if self.current_state is None:
            self.current_state = np.zeros(DIM)

@dataclass
class Setting:
    name: str
    sensory_vectors: Dict[str, np.ndarray]  # visual, auditory, etc.
    mood_vector: np.ndarray
    associated_events: List[str] = field(default_factory=list)

@dataclass 
class PlotPoint:
    phase: StoryPhase
    description: str
    vector: np.ndarray
    required_characters: Set[str]
    consequences: List[str] = field(default_factory=list)

@dataclass
class StoryState:
    """Complete narrative state at any moment"""
    phase: StoryPhase
    tension: float  # 0.0-1.0
    pov_character: Optional[str]
    current_setting: Optional[str]
    active_goals: List[str]
    foreshadowing: List[np.ndarray]  # Seeds for future payoff

# ============================================================
# HDC NARRATIVE ENGINE
# ============================================================

class StoryHDCEngine:
    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.encoder = self._load_encoder()
        
        # Narrative knowledge bases
        self.characters: Dict[str, Character] = {}
        self.settings: Dict[str, Setting] = {}
        self.plot_structure: Dict[StoryPhase, List[PlotPoint]] = defaultdict(list)
        self.episodic_events: List[Dict] = []  # What has happened
        
        # Working memory (active scene)
        self.recent_sentences: deque = deque(maxlen=10)
        self.scene_composite: np.ndarray = np.zeros(dim)
        self.current_pov: Optional[str] = None
        
        # Long-term narrative memory
        self.event_index = faiss.IndexFlatIP(dim)
        self.event_storage: Dict[int, Dict] = {}
        
    def _load_encoder(self):
        if HAS_TRANSFORMERS:
            return SentenceTransformer('intfloat/multilingual-e5-large')
        return None
    
    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode text to sparse bipolar hypervector"""
        if self.encoder:
            emb = self.encoder.encode(text, normalize_embeddings=True)
            v = emb.astype(np.float32)
            if len(v) != self.dim:
                v = np.resize(v, self.dim)
        else:
            v = np.random.randn(self.dim).astype(np.float32)
        
        v = np.sign(v)
        if SPARSITY < 1.0:
            thresh = np.percentile(np.abs(v), 100 * (1 - SPARSITY))
            v[np.abs(v) < thresh] = 0
        
        if normalize:
            v /= np.linalg.norm(v) + 1e-9
        return v
    
    # HDC Primitives
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Associate two concepts (variable binding)"""
        return a * b
    
    def bundle(self, vectors: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """Superposition with saturation control"""
        if not vectors:
            return np.zeros(self.dim)
        weights = weights or [1.0] * len(vectors)
        s = sum(w * v for w, v in zip(weights, vectors))
        # Iterative normalization prevents gray goo
        for _ in range(2):
            s = np.tanh(s)  # Soft saturation
        return s / (np.linalg.norm(s) + 1e-9)
    
    def permute(self, v: np.ndarray, shift: int) -> np.ndarray:
        """Encode sequence position"""
        return np.roll(v, shift)
    
    def probe(self, probe_vec: np.ndarray, target: np.ndarray) -> float:
        """Similarity (cosine)"""
        return float(np.dot(probe_vec, target))

    # ========================================================
    # CHARACTER MANAGEMENT
    # ========================================================
    
    def create_character(self, name: str, traits: List[str], 
                        relationships: Optional[Dict[str, float]] = None) -> Character:
        """Create character with trait hypervectors"""
        trait_vectors = {
            trait: self.encode(trait) for trait in traits
        }
        
        # Character essence = weighted bundle of traits
        essence = self.bundle(list(trait_vectors.values()))
        
        char = Character(
            name=name,
            traits=trait_vectors,
            relationships=relationships or {},
            arc_vector=essence.copy(),  # Starts as current, evolves
            current_state=essence.copy()
        )
        
        self.characters[name] = char
        return char
    
    def update_character_state(self, name: str, event: str, intensity: float = 0.3):
        """Character reacts to event (emotional update)"""
        if name not in self.characters:
            return
            
        char = self.characters[name]
        event_vec = self.encode(event)
        
        # State drifts toward event influence
        char.current_state = self.bundle([
            char.current_state * (1 - intensity),
            event_vec * intensity
        ])
        
        # Check for arc progression
        arc_deviation = self.probe(char.current_state, char.arc_vector)
        if arc_deviation < 0.5:  # Character has changed significantly
            char.arc_vector = char.current_state.copy()
    
    def get_character_voice(self, name: str) -> np.ndarray:
        """Retrieve character's speaking style vector"""
        if name not in self.characters:
            return np.zeros(self.dim)
        
        char = self.characters[name]
        # Voice = current state ⊗ personal traits
        voice = self.bind(char.current_state, 
                         self.bundle(list(char.traits.values())))
        return voice

    # ========================================================
    # SETTING & ATMOSPHERE
    # ========================================================
    
    def create_setting(self, name: str, mood: str, 
                       sensory_details: List[str]) -> Setting:
        """Establish location with sensory hypervectors"""
        sensory = {}
        for detail in sensory_details:
            # Tag with sense type (simple heuristic)
            if any(w in detail for w in ['see', 'look', 'color', 'light']):
                tag = 'visual'
            elif any(w in detail for w in ['hear', 'sound', 'noise', 'silent']):
                tag = 'auditory'
            elif any(w in detail for w in ['smell', 'scent', 'odor']):
                tag = 'olfactory'
            else:
                tag = 'general'
            
            vec = self.encode(detail)
            sensory[tag] = vec  # Keep last of each type
        
        setting = Setting(
            name=name,
            sensory_vectors=sensory,
            mood_vector=self.encode(mood)
        )
        self.settings[name] = setting
        return setting
    
    def get_atmosphere(self, setting_name: str, focus_sense: Optional[str] = None) -> np.ndarray:
        """Retrieve setting atmosphere, optionally emphasizing one sense"""
        if setting_name not in self.settings:
            return np.zeros(self.dim)
        
        s = self.settings[setting_name]
        vectors = [s.mood_vector] + list(s.sensory_vectors.values())
        
        if focus_sense and focus_sense in s.sensory_vectors:
            # Boost specific sense
            weights = [0.3] + [0.7 if k == focus_sense else 0.1 
                             for k in s.sensory_vectors.keys()]
            return self.bundle(vectors, weights)
        
        return self.bundle(vectors)

    # ========================================================
    # PLOT STRUCTURE ENGINE
    # ========================================================
    
    def define_plot_point(self, phase: StoryPhase, description: str,
                         required_chars: List[str], consequences: List[str]):
        """Add structural requirement to story"""
        vec = self.encode(description)
        
        pp = PlotPoint(
            phase=phase,
            description=description,
            vector=vec,
            required_characters=set(required_chars),
            consequences=consequences
        )
        
        self.plot_structure[phase].append(pp)
        
        # Index for retrieval
        self.event_index.add(vec.astype('float32').reshape(1, -1))
        idx = len(self.event_storage)
        self.event_storage[idx] = {
            'type': 'plot_point',
            'data': pp,
            'vector': vec
        }
    
    def get_narrative_drive(self, current_phase: StoryPhase, 
                           tension: float) -> np.ndarray:
        """What should happen next given story structure?"""
        # Retrieve relevant plot points
        phase_points = self.plot_structure.get(current_phase, [])
        
        if not phase_points:
            return np.zeros(self.dim)
        
        # Select based on tension match
        if current_phase in [StoryPhase.RISING_ACTION, StoryPhase.CLIMAX]:
            # Higher tension = more dramatic plot points
            target_vec = self.encode("dramatic conflict escalation")
        else:
            target_vec = self.encode("resolution reflection calm")
        
        # Find nearest plot point
        best = max(phase_points, 
                  key=lambda pp: self.probe(target_vec, pp.vector))
        
        return best.vector

    # ========================================================
    # SENTENCE GENERATION (Core Creative Act)
    # ========================================================
    
    def generate_sentence(self, story_state: StoryState, 
                         sentence_type: str = "narrative") -> str:
        """
        Generate one sentence constrained by:
        - POV character's voice
        - Current setting atmosphere  
        - Narrative drive (what needs to happen)
        - Recent context (working memory)
        """
        components = []
        
        # 1. Character voice (if dialogue or close POV)
        if story_state.pov_character:
            voice = self.get_character_voice(story_state.pov_character)
            components.append((voice, 0.4))
        
        # 2. Setting atmosphere
        if story_state.current_setting:
            atm = self.get_atmosphere(story_state.current_setting)
            components.append((atm, 0.3))
        
        # 3. Narrative drive (plot requirements)
        drive = self.get_narrative_drive(story_state.phase, story_state.tension)
        components.append((drive, 0.2))
        
        # 4. Recent context (continuity)
        if self.recent_sentences:
            context = self.bundle([self.encode(s) for s in self.recent_sentences])
            components.append((context, 0.1))
        
        # Composite query
        query = self.bundle([v for v, _ in components], 
                           weights=[w for _, w in components])
        
        # Retrieve vocabulary (simplified: would use actual word index)
        # Here we simulate with template filling
        return self._fill_template(query, story_state, sentence_type)
    
    def _fill_template(self, query_vec: np.ndarray, 
                       state: StoryState, sent_type: str) -> str:
        """Template-based generation with HDC selection"""
        
        # Sentence structure based on type and tension
        if sent_type == "dialogue":
            templates = [
                "{char} said, '{content}'",
                "'{content}', {char} {verb}",
                "{char} {verb}, '{content}'"
            ]
        elif state.tension > 0.7:
            templates = [
                "{subject} {violent_verb} {object} in {setting}",
                "Suddenly, {subject} {violent_verb}",
                "{setting} erupted in {chaos}"
            ]
        else:
            templates = [
                "{subject} {calm_verb} through {setting}",
                "{setting} was {description}",
                "{subject} noticed {detail}"
            ]
        
        # Select template via similarity to query
        template_vecs = [self.encode(t) for t in templates]
        best_idx = np.argmax([self.probe(query_vec, tv) for tv in template_vecs])
        template = templates[best_idx]
        
        # Fill slots via retrieval
        # (Simplified: would retrieve from indexed vocabulary)
        char = state.pov_character or "the figure"
        setting = state.current_setting or "the place"
        
        # Return filled template
        return template.format(
            char=char,
            setting=setting,
            content="words here",  # Would generate via word-level HDC
            subject=char,
            violent_verb="struck",
            calm_verb="walked",
            description="quiet",
            detail="something",
            chaos="flame",
            verb="whispered"
        )

    # ========================================================
    # SCENE COMPOSITION
    # ========================================================
    
    def compose_scene(self, setting_name: str, pov_char: str,
                      goal: str, target_words: int = 300) -> str:
        """
        Generate complete scene with:
        - Character goal pursuit
        - Setting interaction
        - Rising/falling tension arc
        """
        if setting_name not in self.settings:
            return "[Setting not found]"
        
        # Initialize scene state
        story_state = StoryState(
            phase=StoryPhase.RISING_ACTION,
            tension=0.3,
            pov_character=pov_char,
            current_setting=setting_name,
            active_goals=[goal],
            foreshadowing=[]
        )
        
        sentences = []
        word_count = 0
        
        while word_count < target_words:
            # Determine sentence type
            if len(sentences) % 5 == 0 and len(sentences) > 0:
                sent_type = "dialogue"  # Periodic dialogue
            else:
                sent_type = "narrative"
            
            # Generate
            sent = self.generate_sentence(story_state, sent_type)
            sentences.append(sent)
            word_count += len(sent.split())
            
            # Update working memory
            self.recent_sentences.append(sent)
            
            # Update state (tension escalates toward middle, resolves)
            progress = word_count / target_words
            if progress < 0.6:
                story_state.tension = min(0.9, story_state.tension + 0.05)
            else:
                story_state.tension = max(0.1, story_state.tension - 0.08)
            
            # Check for phase transition
            if progress > 0.8 and story_state.phase == StoryPhase.RISING_ACTION:
                story_state.phase = StoryPhase.CLIMAX
            
            # Character state updates based on events
            event_content = sent[:50]  # Simplified event extraction
            self.update_character_state(pov_char, event_content)
        
        # Store scene in episodic memory
        scene_text = " ".join(sentences)
        scene_vec = self.encode(scene_text)
        self._store_event("scene", scene_text, scene_vec, 
                         characters={pov_char}, setting=setting_name)
        
        return scene_text
    
    def _store_event(self, event_type: str, content: str, 
                     vector: np.ndarray, characters: Set[str], setting: str):
        """Index event for future retrieval"""
        idx = len(self.episodic_events)
        self.episodic_events.append({
            'type': event_type,
            'content': content,
            'vector': vector,
            'characters': characters,
            'setting': setting
        })
        self.event_index.add(vector.astype('float32').reshape(1, -1))

    # ========================================================
    # CHAPTER & LONG-FORM ASSEMBLY
    # ========================================================
    
    def compose_chapter(self, title: str, scenes_config: List[Dict]) -> str:
        """
        Assemble chapter from multiple scenes with continuity
        """
        chapter_parts = [f"# {title}\n"]
        
        for i, config in enumerate(scenes_config):
            print(f"Composing scene {i+1}/{len(scenes_config)}...")
            
            scene_text = self.compose_scene(
                setting_name=config['setting'],
                pov_char=config['pov'],
                goal=config['goal'],
                target_words=config.get('words', 300)
            )
            
            chapter_parts.append(f"\n## Scene {i+1}: {config['setting']}\n")
            chapter_parts.append(scene_text)
            
            # Scene transition: update characters
            for char_name in config.get('involved_chars', []):
                if char_name in self.characters:
                    # Character carries forward
                    pass
        
        return "\n".join(chapter_parts)
    
    def generate_story(self, premise: str, chapters: int = 3) -> str:
        """
        Full story generation with narrative arc
        """
        print(f"Generating story: {premise}")
        
        # Encode premise as initial condition
        premise_vec = self.encode(premise)
        
        # Define characters from premise (simplified)
        chars = self._extract_characters(premise)
        for char_name, traits in chars.items():
            self.create_character(char_name, traits)
        
        # Define settings
        settings = self._extract_settings(premise)
        for setting_name, mood in settings.items():
            self.create_setting(setting_name, mood, 
                              [f"{setting_name} detail {i}" for i in range(3)])
        
        # Define plot structure
        self._create_plot_arc(premise)
        
        # Generate chapters
        full_story = [f"Story: {premise}\n"]
        
        for ch_num in range(chapters):
            phase = self._phase_for_chapter(ch_num, chapters)
            scenes = self._scenes_for_phase(phase, chars, settings)
            
            chapter = self.compose_chapter(
                title=f"Chapter {ch_num + 1}: {phase.name.replace('_', ' ').title()}",
                scenes_config=scenes
            )
            full_story.append(chapter)
        
        return "\n\n---\n\n".join(full_story)
    
    # Helper methods (simplified implementations)
    def _extract_characters(self, premise: str) -> Dict[str, List[str]]:
        # Would use NER or pattern matching
        return {"Rama": ["brave", "conflicted"], "Sita": ["wise", "determined"]}
    
    def _extract_settings(self, premise: str) -> Dict[str, str]:
        return {"Forest": "mysterious", "Palace": "oppressive"}
    
    def _create_plot_arc(self, premise: str):
        # Define 3-act structure
        self.define_plot_point(StoryPhase.EXPOSITION, 
                              "Hero discovers conflict", ["Rama"], 
                              ["Hero commits to journey"])
        self.define_plot_point(StoryPhase.RISING_ACTION,
                              "Hero faces first test", ["Rama", "Sita"],
                              ["Alliance formed", "Stakes raised"])
        self.define_plot_point(StoryPhase.CLIMAX,
                              "Final confrontation", ["Rama"],
                              ["Resolution achieved"])
    
    def _phase_for_chapter(self, ch_num: int, total: int) -> StoryPhase:
        ratio = ch_num / total
        if ratio < 0.2: return StoryPhase.EXPOSITION
        elif ratio < 0.6: return StoryPhase.RISING_ACTION
        elif ratio < 0.8: return StoryPhase.CLIMAX
        else: return StoryPhase.RESOLUTION
    
    def _scenes_for_phase(self, phase: StoryPhase, chars, settings):
        # Return scene configurations
        return [
            {'setting': 'Forest', 'pov': 'Rama', 'goal': 'Find path', 
             'words': 250, 'involved_chars': ['Rama']},
            {'setting': 'Palace', 'pov': 'Sita', 'goal': 'Gather information',
             'words': 300, 'involved_chars': ['Sita', 'Rama']}
        ]

# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    print("GodNode-Story: Hierarchical HDC Narrative Generation")
    print("=" * 60)
    
    engine = StoryHDCEngine()
    
    # Generate complete story
    story = engine.generate_story(
        premise="A warrior prince exiled to the forest must rescue his wife " +
                "from a demon king, discovering that true strength lies in compassion",
        chapters=3
    )
    
    print(story)
    
    # Demonstrate character consistency
    print("\n" + "=" * 60)
    print("CHARACTER STATE CHECK:")
    for name, char in engine.characters.items():
        print(f"\n{name}:")
        print(f"  Initial traits: {list(char.traits.keys())}")
        print(f"  Current state deviation: {engine.probe(char.current_state, char.arc_vector):.3f}")
