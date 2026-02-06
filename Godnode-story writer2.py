"""
GodNode-Story v3.1 - Narrative Extension Module
Adds: Character consistency, plot arcs, scene management, dialogue attribution
Compatible with GodNode v3.0 core
"""

import numpy as np
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import json

# ═══════════════════════════════════════════════════════════════════════════
# NARRATIVE DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

class NarrativePhase(Enum):
    EXPOSITION = auto()
    RISING_ACTION = auto()
    CLIMAX = auto()
    FALLING_ACTION = auto()
    RESOLUTION = auto()

@dataclass
class CharacterState:
    """Dynamic character representation"""
    name: str
    traits: Dict[str, np.ndarray]  # Static personality
    emotional_state: np.ndarray    # Current feelings
    relationships: Dict[str, float]  # Bond strength to others
    arc_progress: float = 0.0      # 0=start, 1=transformed
    appearances: int = 0
    
    def get_voice_vector(self, engine) -> np.ndarray:
        """Character's unique speaking style"""
        # Blend traits with current emotion
        trait_bundle = engine.bundle(list(self.traits.values()))
        return engine.bundle([
            trait_bundle * 0.7,
            self.emotional_state * 0.3
        ])

@dataclass
class StoryBeat:
    """Plot point with preconditions and effects"""
    phase: NarrativePhase
    description: str
    vector: np.ndarray
    required_characters: Set[str]
    tension_level: float  # 0.0-1.0
    consequences: List[str] = field(default_factory=list)
    fulfilled: bool = False

@dataclass
class SceneState:
    """Runtime scene composition state"""
    setting: str
    pov_character: str
    present_characters: Set[str]
    tension: float
    phase: NarrativePhase
    goal: str
    beats_completed: List[str] = field(default_factory=list)

# ═══════════════════════════════════════════════════════════════════════════
# STORY GENERATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class GodNodeStory(GodNode):
    """Extends GodNode v3.0 with narrative capabilities"""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Narrative knowledge bases
        self.characters: Dict[str, CharacterState] = {}
        self.settings: Dict[str, np.ndarray] = {}
        self.plot_beats: Dict[NarrativePhase, List[StoryBeat]] = {
            phase: [] for phase in NarrativePhase
        }
        self.scene_history: deque = deque(maxlen=50)  # Recent scenes
        
        # Generation state
        self.current_phase = NarrativePhase.EXPOSITION
        self.global_tension = 0.3
        
    # ═══════════════════════════════════════════════════════════════════════
    # CHARACTER MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════
    
    def create_character(self, name: str, traits: List[str], 
                        relationships: Optional[Dict[str, float]] = None):
        """Initialize character with trait hypervectors"""
        trait_vecs = {
            trait: self.vec(trait) for trait in traits
        }
        
        # Initial emotional state = average of traits
        emotion = self.bundle(list(trait_vecs.values()))
        
        char = CharacterState(
            name=name,
            traits=trait_vecs,
            emotional_state=emotion,
            relationships=relationships or {}
        )
        
        self.characters[name] = char
        return char
    
    def update_character_emotion(self, name: str, event: str, intensity: float = 0.3):
        """Character reacts to story event"""
        if name not in self.characters:
            return
            
        char = self.characters[name]
        event_vec = self.vec(event)
        
        # Emotional state drifts toward event
        char.emotional_state = self.bundle([
            char.emotional_state * (1 - intensity),
            event_vec * intensity
        ])
        
        # Track arc progression
        initial = self.bundle(list(char.traits.values()))
        deviation = 1 - self.cosine(char.emotional_state, initial)
        char.arc_progress = min(1.0, deviation * 2)  # Scale to 0-1
    
    def get_character_consistency_check(self, name: str, action: str) -> float:
        """Verify action is in-character (0-1 score)"""
        if name not in self.characters:
            return 0.5
            
        char = self.characters[name]
        action_vec = self.vec(action)
        
        # Check trait alignment
        trait_sims = [self.cosine(action_vec, t) for t in char.traits.values()]
        best_trait_match = max(trait_sims) if trait_sims else 0
        
        # Check emotional consistency
        emotion_sim = self.cosine(action_vec, char.emotional_state)
        
        # Weight: traits matter more early, emotion matters more late
        trait_weight = 1 - char.arc_progress
        score = trait_weight * best_trait_match + (1 - trait_weight) * emotion_sim
        
        return float(score)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PLOT STRUCTURE
    # ═══════════════════════════════════════════════════════════════════════
    
    def define_plot_beat(self, phase: NarrativePhase, description: str,
                        required_chars: List[str], tension: float,
                        consequences: List[str]):
        """Add structural requirement to narrative"""
        vec = self.vec(description)
        
        beat = StoryBeat(
            phase=phase,
            description=description,
            vector=vec,
            required_characters=set(required_chars),
            tension_level=tension,
            consequences=consequences
        )
        
        self.plot_beats[phase].append(beat)
    
    def get_next_required_beat(self) -> Optional[StoryBeat]:
        """What plot point must happen next?"""
        candidates = self.plot_beats[self.current_phase]
        unfulfilled = [b for b in candidates if not b.fulfilled]
        
        if not unfulfilled:
            # Advance phase
            self._advance_phase()
            return self.get_next_required_beat()
        
        # Select by tension match
        target_tension = self.global_tension
        best = min(unfulfilled, 
                  key=lambda b: abs(b.tension_level - target_tension))
        
        return best
    
    def _advance_phase(self):
        """Move to next narrative phase"""
        phases = list(NarrativePhase)
        idx = phases.index(self.current_phase)
        if idx < len(phases) - 1:
            self.current_phase = phases[idx + 1]
            print(f"  → Advancing to {self.current_phase.name}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # SCENE COMPOSITION
    # ═══════════════════════════════════════════════════════════════════════
    
    def compose_scene(self, setting: str, pov: str, 
                     goal: str, target_words: int = 400) -> Dict:
        """
        Generate complete scene with narrative coherence
        """
        if pov not in self.characters:
            raise ValueError(f"Character {pov} not defined")
        
        # Initialize scene state
        scene = SceneState(
            setting=setting,
            pov_character=pov,
            present_characters={pov},  # Start with POV character
            tension=self.global_tension,
            phase=self.current_phase,
            goal=goal
        )
        
        paragraphs = []
        word_count = 0
        
        # Get required plot beat
        required_beat = self.get_next_required_beat()
        if required_beat:
            print(f"  Required beat: {required_beat.description[:50]}...")
        
        # Generate paragraphs until target length
        while word_count < target_words:
            para = self._compose_paragraph(scene, required_beat)
            paragraphs.append(para['text'])
            word_count += len(para['text'].split())
            
            # Update scene state
            scene.tension = para['ending_tension']
            self.update_character_emotion(pov, para['key_event'], 0.2)
            
            # Check if beat fulfilled
            if required_beat and para['fulfills_beat']:
                required_beat.fulfilled = True
                scene.beats_completed.append(required_beat.description)
                required_beat = self.get_next_required_beat()
        
        # Store scene
        scene_record = {
            'setting': setting,
            'pov': pov,
            'goal': goal,
            'paragraphs': paragraphs,
            'word_count': word_count,
            'final_tension': scene.tension,
            'beats_completed': scene.beats_completed,
            'vector': self.bundle([self.vec(p) for p in paragraphs])
        }
        self.scene_history.append(scene_record)
        
        # Update global state
        self.global_tension = scene.tension
        
        return scene_record
    
    def _compose_paragraph(self, scene: SceneState, 
                          target_beat: Optional[StoryBeat]) -> Dict:
        """Generate one paragraph with narrative constraints"""
        
        pov_char = self.characters[scene.pov_character]
        
        # Build compositional query
        components = []
        
        # 1. POV character's voice (strong weight)
        voice = pov_char.get_voice_vector(self)
        components.append((voice, 0.35))
        
        # 2. Setting atmosphere
        if scene.setting in self.settings:
            components.append((self.settings[scene.setting], 0.20))
        
        # 3. Plot requirement (if exists)
        if target_beat:
            components.append((target_beat.vector, 0.25))
        
        # 4. Recent context (continuity)
        if self.scene_history:
            recent = self.bundle([s['vector'] for s in list(self.scene_history)[-3:]])
            components.append((recent, 0.15))
        
        # 5. Tension modulation
        tension_vec = self.vec(f"tension_{int(scene.tension * 10)}")
        components.append((tension_vec, 0.05))
        
        # Composite query
        query = self.bundle([v for v, _ in components])
        
        # Generate via HDC (simplified: would use word-level generation)
        # Here we simulate with structured output
        fulfills = False
        if target_beat and np.random.random() < 0.3:
            # Include plot beat content
            text = f"{scene.pov_character} faced {target_beat.description}. "
            fulfills = True
        else:
            # Atmospheric description
            text = f"The {scene.setting} surrounded {scene.pov_character}. "
        
        # Add dialogue if characters present
        others = scene.present_characters - {scene.pov_character}
        if others and np.random.random() < 0.4:
            other = np.random.choice(list(others))
            text += f"'What do you want?' {scene.pov_character} asked. "
        
        # Determine tension change
        tension_delta = np.random.uniform(-0.1, 0.15)
        new_tension = np.clip(scene.tension + tension_delta, 0.0, 1.0)
        
        return {
            'text': text,
            'ending_tension': new_tension,
            'fulfills_beat': fulfills,
            'key_event': text[:50]
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # DIALOGUE GENERATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def compose_dialogue(self, speaker: str, listener: str, 
                        context: str, emotional_tone: str) -> str:
        """
        Generate dialogue attributed to specific character
        """
        if speaker not in self.characters:
            return f"[Unknown character: {speaker}]"
        
        char = self.characters[speaker]
        
        # Build speaker-specific query
        speaker_voice = char.get_voice_vector(self)
        tone_vec = self.vec(emotional_tone)
        context_vec = self.vec(context)
        
        # Bind speaker to tone to context
        query = self.bind(speaker_voice, self.bind(tone_vec, context_vec))
        
        # Retrieve dialogue-appropriate phrases
        # (Would search specialized dialogue vocabulary)
        candidates = self.search(query, k=5)
        
        # Filter by character consistency
        valid = []
        for phrase, sim in candidates:
            consistency = self.get_character_consistency_check(speaker, phrase)
            if consistency > 0.6:  # Threshold for in-character
                valid.append((phrase, sim * consistency))
        
        if valid:
            return max(valid, key=lambda x: x[1])[0]
        
        return f"[{speaker} speaks in {emotional_tone} tone about {context}]"
    
    # ═══════════════════════════════════════════════════════════════════════
    # FULL STORY GENERATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def generate_story(self, premise: str, chapters: int = 3,
                      scenes_per_chapter: int = 3) -> Dict:
        """
        Generate complete story with narrative arc
        """
        print(f"\n{'='*70}")
        print(f"GENERATING STORY: {premise[:60]}...")
        print(f"{'='*70}")
        
        # Setup from premise
        self._setup_from_premise(premise)
        
        story = {
            'title': premise.split('.')[0],
            'premise': premise,
            'chapters': []
        }
        
        for ch_num in range(1, chapters + 1):
            print(f"\n--- Chapter {ch_num}: {self.current_phase.name} ---")
            
            chapter = {
                'number': ch_num,
                'phase': self.current_phase.name,
                'scenes': []
            }
            
            for sc_num in range(1, scenes_per_chapter + 1):
                # Select POV (rotate or fixed)
                pov = np.random.choice(list(self.characters.keys()))
                setting = np.random.choice(list(self.settings.keys()))
                goal = f"advance_{self.current_phase.name.lower()}"
                
                print(f"  Scene {sc_num}: {pov} in {setting}")
                
                scene = self.compose_scene(setting, pov, goal, target_words=300)
                chapter['scenes'].append(scene)
            
            story['chapters'].append(chapter)
            
            # Phase transition between chapters
            if ch_num < chapters:
                self._advance_phase()
        
        print(f"\n{'='*70}")
        print("STORY COMPLETE")
        print(f"{'='*70}")
        
        return story
    
    def _setup_from_premise(self, premise: str):
        """Extract and initialize story elements from premise"""
        premise_vec = self.vec(premise)
        
        # Search for similar stories to infer structure
        # (Would use case-based reasoning in full system)
        
        # Default setup if no prior knowledge
        if not self.characters:
            # Create protagonist from premise
            self.create_character("protagonist", ["brave", "conflicted"])
            self.create_character("antagonist", ["powerful", "ruthless"])
        
        if not self.settings:
            self.settings["forest"] = self.vec("mysterious forest")
            self.settings["village"] = self.vec("quiet village")
        
        # Define default plot
        if not any(self.plot_beats.values()):
            self.define_plot_beat(
                NarrativePhase.EXPOSITION,
                "hero discovers ordinary world",
                ["protagonist"], 0.2,
                ["hero commits to journey"]
            )
            self.define_plot_beat(
                NarrativePhase.RISING_ACTION,
                "hero faces increasing challenges",
                ["protagonist", "antagonist"], 0.6,
                ["conflict escalates"]
            )
            self.define_plot_beat(
                NarrativePhase.CLIMAX,
                "final confrontation with antagonist",
                ["protagonist", "antagonist"], 0.9,
                ["resolution begins"]
            )
    
    def format_story(self, story: Dict) -> str:
        """Pretty print story"""
        lines = [
            f"# {story['title']}",
            f"\nPremise: {story['premise']}\n"
        ]
        
        for ch in story['chapters']:
            lines.append(f"## Chapter {ch['number']}: {ch['phase']}")
            
            for i, sc in enumerate(ch['scenes'], 1):
                lines.append(f"\n### Scene {i}")
                lines.append(f"Setting: {sc['setting']} | POV: {sc['pov']}")
                lines.append(f"Beats: {', '.join(sc['beats_completed']) or 'None'}")
                lines.append(f"\n{' '.join(sc['paragraphs'][:2])}...")  # First 2 paras
        
        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════

def demo_story_generation():
    """Demonstrate story generation capabilities"""
    
    # Use GodNode v3.0 config
    config = GodNodeConfig(
        dim=1024,
        sparsity=0.05,
        poetry_length=20,
        beam_width=10
    )
    
    story_engine = GodNodeStory(config)
    
    # Train on narrative corpus
    corpus = """
    The warrior stood at the forest edge, his sword heavy with memory.
    'You cannot pass,' the shadow spoke, and the air grew cold.
    She laughed, though fear chilled her heart. 'Watch me.'
    The village burned behind them, smoke rising like accusation.
    In silence, they walked toward the mountain, destiny pressing close.
    """ * 50
    
    # Write temp corpus
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(corpus)
        corpus_path = f.name
    
    story_engine.train_on_corpus([corpus_path])
    os.remove(corpus_path)
    
    # Define specific story
    story_engine.create_character("Rama", ["exiled", "righteous", "skilled"])
    story_engine.create_character("Sita", ["loyal", "wise", "resilient"])
    story_engine.create_character("Ravana", ["powerful", "learned", "proud"])
    
    story_engine.settings["forest"] = story_engine.vec("ancient forest mystery")
    story_engine.settings["palace"] = story_engine.vec("golden palace oppression")
    story_engine.settings["mountain"] = story_engine.vec("sacred mountain trials")
    
    story_engine.define_plot_beat(
        NarrativePhase.EXPOSITION,
        "Rama accepts exile to forest",
        ["Rama", "Sita"], 0.3,
        ["Sita joins Rama", "peace begins"]
    )
    story_engine.define_plot_beat(
        NarrativePhase.RISING_ACTION,
        "Sita is taken from forest",
        ["Rama", "Sita", "Ravana"], 0.7,
        ["Rama pursues", "alliance forms"]
    )
    story_engine.define_plot_beat(
        NarrativePhase.CLIMAX,
        "Battle at gates of Lanka",
        ["Rama", "Ravana"], 0.95,
        ["dharma tested", "transformation"]
    )
    
    # Generate
    story = story_engine.generate_story(
        premise="An exiled prince must rescue his wife from a demon king, "
                "discovering that true strength lies in compassion, not power.",
        chapters=3,
        scenes_per_chapter=2
    )
    
    print(story_engine.format_story(story))
    
    # Test dialogue
    print("\n" + "="*70)
    print("SAMPLE DIALOGUE:")
    print("="*70)
    
    dialogue = story_engine.compose_dialogue(
        speaker="Rama",
        listener="Ravana",
        context="final confrontation before battle",
        emotional_tone="determined sorrow"
    )
    print(f"Rama: '{dialogue}'")
    
    # Consistency check
    consistency = story_engine.get_character_consistency_check(
        "Rama", "offers mercy to defeated enemy"
    )
    print(f"\nConsistency check (mercy offer): {consistency:.2f}")


if __name__ == "__main__":
    demo_story_generation()
