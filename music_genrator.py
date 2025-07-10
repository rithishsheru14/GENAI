import numpy as np
from typing import List, Dict, Optional, Set
import json
import os
from dataclasses import dataclass
import midiutil
import random
from transformers import pipeline
import spacy
import torch
import re

@dataclass
class MusicParameters:
    tempo: int
    key: str
    scale: str
    time_signature: tuple
    mood: str
    theme: str
    instruments: List[Dict[str, any]]
    duration: int
    genre: Optional[str]
    intensity: float  # 0.0 to 1.0
    complexity: float  # 0.0 to 1.0

class InstrumentMapper:
    def __init__(self):
        # Comprehensive MIDI instrument mapping
        self.instrument_map = {
            # Piano family
            "piano": 0, "grand piano": 0, "upright piano": 1, "electric piano": 4,
            "keyboard": 0, "synthesizer": 80,
            
            # String instruments
            "violin": 40, "viola": 41, "cello": 42, "contrabass": 43,
            "guitar": 24, "acoustic guitar": 25, "electric guitar": 27,
            "bass": 32, "acoustic bass": 32, "electric bass": 33,
            "orchestral strings": 48,
            
            # Wind instruments
            "flute": 73, "piccolo": 72, "clarinet": 71, "oboe": 68,
            "saxophone": 65, "trumpet": 56, "trombone": 57, "french horn": 60,
            
            # Percussion
            "drums": 0, "drum kit": 0, "percussion": 0, "cymbals": 0,
            "timpani": 47, "xylophone": 13, "marimba": 12,
            
            # Electronic
            "synth": 80, "pad": 89, "atmosphere": 99, "fx": 102,
            "electronic": 81
        }
        
        # Instrument characteristics
        self.instrument_traits = {
            "piano": {"melodic": True, "harmonic": True, "percussive": True, "range": "wide"},
            "violin": {"melodic": True, "harmonic": False, "sustain": True, "range": "high"},
            "drums": {"melodic": False, "harmonic": False, "percussive": True, "range": "fixed"},
            # Add more instrument characteristics
        }

    def get_midi_instrument(self, instrument_name: str) -> int:
        """Get MIDI program number for instrument name."""
        # Clean and normalize instrument name
        instrument_name = instrument_name.lower().strip()
        
        # Try direct match
        if instrument_name in self.instrument_map:
            return self.instrument_map[instrument_name]
        
        # Try partial match
        for key in self.instrument_map:
            if key in instrument_name or instrument_name in key:
                return self.instrument_map[key]
        
        # Default to piano if no match found
        return 0

class PromptAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment = pipeline("sentiment-analysis")
        self.instrument_mapper = InstrumentMapper()
        
        # Musical context mappings
        self.mood_mappings = {
            "happy": {
                "scales": ["C major", "G major", "D major"],
                "tempo_range": (120, 160),
                "intensity": 0.7,
                "note_lengths": [0.25, 0.5, 1.0]
            },
            "sad": {
                "scales": ["A minor", "D minor", "E minor"],
                "tempo_range": (60, 80),
                "intensity": 0.4,
                "note_lengths": [1.0, 2.0, 4.0]
            },
            "epic": {
                "scales": ["C minor", "G minor", "D minor"],
                "tempo_range": (90, 120),
                "intensity": 0.9,
                "note_lengths": [0.5, 1.0, 2.0]
            },
            "peaceful": {
                "scales": ["F major", "Bb major", "G major"],
                "tempo_range": (70, 90),
                "intensity": 0.3,
                "note_lengths": [1.0, 2.0, 4.0]
            },
            "angry": {
                "scales": ["E minor", "B minor", "F# minor"],
                "tempo_range": (140, 180),
                "intensity": 0.9,
                "note_lengths": [0.125, 0.25, 0.5]
            }
        }
        
        self.genre_patterns = {
            "classical": {"complexity": 0.8, "instruments": ["piano", "violin", "cello"]},
            "rock": {"complexity": 0.6, "instruments": ["electric guitar", "bass", "drums"]},
            "jazz": {"complexity": 0.9, "instruments": ["piano", "saxophone", "bass"]},
            "electronic": {"complexity": 0.5, "instruments": ["synth", "drums", "bass"]},
            "ambient": {"complexity": 0.3, "instruments": ["pad", "atmosphere", "synth"]}
        }

    def extract_instruments(self, text: str) -> List[Dict[str, any]]:
        """Extract instruments and their roles from text."""
        doc = self.nlp(text.lower())
        instruments = []
        
        # Look for instrument mentions and their contexts
        for token in doc:
            if token.text in self.instrument_mapper.instrument_map:
                # Analyze context for role and style
                preceding_words = [t.text for t in token.lefts]
                following_words = [t.text for t in token.rights]
                
                instrument = {
                    "name": token.text,
                    "midi_program": self.instrument_mapper.get_midi_instrument(token.text),
                    "role": self._determine_role(preceding_words, following_words),
                    "style": self._determine_style(preceding_words, following_words)
                }
                instruments.append(instrument)
        
        # Add default instruments if none found
        if not instruments:
            instruments = [
                {"name": "piano", "midi_program": 0, "role": "main", "style": "normal"},
                {"name": "bass", "midi_program": 32, "role": "backing", "style": "normal"},
                {"name": "drums", "midi_program": 0, "role": "rhythm", "style": "normal"}
            ]
        
        return instruments

    def _determine_role(self, preceding_words: List[str], following_words: List[str]) -> str:
        """Determine instrument's role based on context."""
        role_keywords = {
            "main": ["lead", "melody", "solo", "main"],
            "backing": ["accompaniment", "backing", "background", "harmony"],
            "rhythm": ["rhythm", "beat", "groove"],
            "bass": ["bass", "low"]
        }
        
        context_words = set(preceding_words + following_words)
        
        for role, keywords in role_keywords.items():
            if any(word in context_words for word in keywords):
                return role
        
        return "main"  # default role

    def _determine_style(self, preceding_words: List[str], following_words: List[str]) -> str:
        """Determine playing style based on context."""
        style_keywords = {
            "aggressive": ["aggressive", "heavy", "hard", "intense"],
            "gentle": ["gentle", "soft", "light", "delicate"],
            "staccato": ["sharp", "staccato", "punctuated"],
            "legato": ["smooth", "flowing", "legato"],
            "tremolo": ["tremolo", "shaking", "vibrating"]
        }
        
        context_words = set(preceding_words + following_words)
        
        for style, keywords in style_keywords.items():
            if any(word in context_words for word in keywords):
                return style
        
        return "normal"

    def analyze_prompt(self, prompt: str) -> MusicParameters:
        """Analyze prompt for musical parameters."""
        doc = self.nlp(prompt.lower())
        
        # Extract mood
        sentiment = self.sentiment(prompt)[0]
        base_mood = "happy" if sentiment["label"] == "POSITIVE" else "sad"
        
        # Look for specific mood overrides
        for mood in self.mood_mappings:
            if mood in prompt.lower():
                base_mood = mood
        
        # Extract theme and genre
        theme = self._extract_theme(doc)
        genre = self._extract_genre(doc)
        
        # Get instruments with their roles
        instruments = self.extract_instruments(prompt)
        
        # Determine musical parameters based on mood and genre
        mood_params = self.mood_mappings[base_mood]
        
        # Adjust complexity based on instruments and genre
        complexity = self._calculate_complexity(instruments, genre)
        
        return MusicParameters(
            tempo=random.randint(*mood_params["tempo_range"]),
            key=random.choice(mood_params["scales"]),
            scale="major" if "major" in mood_params["scales"][0] else "minor",
            time_signature=self._determine_time_signature(genre),
            mood=base_mood,
            theme=theme,
            instruments=instruments,
            duration=30,
            genre=genre,
            intensity=mood_params["intensity"],
            complexity=complexity
        )

    def _extract_theme(self, doc) -> str:
        """Extract theme from prompt."""
        theme_keywords = {
            "nature": ["forest", "ocean", "mountain", "river", "wind"],
            "urban": ["city", "street", "urban", "downtown"],
            "space": ["space", "stars", "galaxy", "cosmic"],
            "emotional": ["love", "heartbreak", "joy", "sorrow"]
        }
        
        text = doc.text.lower()
        for theme, keywords in theme_keywords.items():
            if any(keyword in text for keyword in keywords):
                return theme
        
        return "general"

    def _extract_genre(self, doc) -> Optional[str]:
        """Extract musical genre from prompt."""
        text = doc.text.lower()
        for genre in self.genre_patterns:
            if genre in text:
                return genre
        return None

    def _calculate_complexity(self, instruments: List[Dict], genre: Optional[str]) -> float:
        """Calculate musical complexity based on instruments and genre."""
        base_complexity = 0.5
        
        # Adjust for genre
        if genre and genre in self.genre_patterns:
            base_complexity = self.genre_patterns[genre]["complexity"]
        
        # Adjust for number and types of instruments
        instrument_count_factor = min(len(instruments) / 4, 1.0)
        base_complexity += instrument_count_factor * 0.2
        
        return min(base_complexity, 1.0)

    def _determine_time_signature(self, genre: Optional[str]) -> tuple:
        """Determine time signature based on genre and context."""
        if genre == "jazz":
            return random.choice([(4, 4), (3, 4), (5, 4)])
        elif genre == "classical":
            return random.choice([(4, 4), (3, 4), (6, 8)])
        elif genre == "electronic":
            return (4, 4)
        return (4, 4)

class MelodyGenerator:
    def __init__(self):
        self.scales = {
            "C major": [60, 62, 64, 65, 67, 69, 71, 72],
            "G major": [67, 69, 71, 72, 74, 76, 78, 79],
            "D major": [62, 64, 66, 67, 69, 71, 73, 74],
            "A minor": [57, 59, 60, 62, 64, 65, 67, 69],
            "E minor": [64, 66, 67, 69, 71, 72, 74, 76],
            # Add more scales
        }
    
    def generate_for_instrument(self, instrument: Dict, params: MusicParameters) -> List[Dict]:
        """Generate melody for specific instrument considering its role and style."""
        notes = []
        current_time = 0
        scale = self.scales.get(params.key, self.scales["C major"])
        
        # Adjust note generation based on instrument role
        if instrument["role"] == "main":
            notes = self._generate_melody_line(scale, params)
        elif instrument["role"] == "backing":
            notes = self._generate_harmony(scale, params)
        elif instrument["role"] == "rhythm":
            notes = self._generate_rhythm(params)
        elif instrument["role"] == "bass":
            notes = self._generate_bass_line(scale, params)
        
        # Apply style modifications
        notes = self._apply_style(notes, instrument["style"], params)
        
        return notes

    def _generate_melody_line(self, scale: List[int], params: MusicParameters) -> List[Dict]:
        """Generate main melody line."""
        notes = []
        current_time = 0
        
        while current_time < params.duration:
            # Use complexity to determine note patterns
            if params.complexity > 0.7:
                # More complex melodic patterns
                pattern_length = random.choice([3, 4, 5])
                pattern = [random.choice(scale) for _ in range(pattern_length)]
                for note_pitch in pattern:
                    duration = random.choice([0.25, 0.5]) if params.intensity > 0.6 else random.choice([0.5, 1.0])
                    notes.append({
                        "pitch": note_pitch,
                        "duration": duration,
                        "velocity": random.randint(70, 100),
                        "time": current_time
                    })
                    current_time += duration
            else:
                # Simpler melodic patterns
                note = {
                    "pitch": random.choice(scale),
                    "duration": random.choice([0.5, 1.0, 2.0]),
                    "velocity": random.randint(60, 90),
                    "time": current_time
                }
                notes.append(note)
                current_time += note["duration"]
        
        return notes

    def _generate_harmony(self, scale: List[int], params: MusicParameters) -> List[Dict]:
        """Generate harmonic accompaniment."""
        notes = []
        current_time = 0
        
        # Create chord progressions based on scale
        chord_roots = [scale[0], scale[3], scale[4], scale[5]]  # I, IV, V, VI
        
        while current_time < params.duration:
            chord_root = random.choice(chord_roots)
            # Build chord (root, third, fifth)
            chord_notes = [chord_root, chord_root + 4, chord_root + 7]
            
            duration = 2.0 if params.intensity < 0.5 else 1.0
            
            for note_pitch in chord_notes:
                notes.append({
                    "pitch": note_pitch,
                    "duration": duration,
                    "velocity": random.randint(40, 60),
                    "time": current_time
                })
            
            current_time += duration
        
        return notes

    def _generate_rhythm(self, params: MusicParameters) -> List[Dict]:
        """Generate rhythm patterns based on genre and intensity."""
        notes = []
        current_time = 0
        
        # Define different rhythm patterns based on genre and mood
        patterns = {
            "rock": [
                {"pitch": 36, "duration": 1.0},  # Bass drum
                {"pitch": 42, "duration": 0.5},  # Closed hi-hat
                {"pitch": 38, "duration": 1.0},  # Snare
                {"pitch": 42, "duration": 0.5}   # Closed hi-hat
            ],
            "jazz": [
                {"pitch": 42, "duration": 0.25},  # Ride cymbal
                {"pitch": 38, "duration": 0.5},   # Snare
                {"pitch": 42, "duration": 0.25},  # Ride cymbal
                {"pitch": 36, "duration": 0.5}    # Bass drum
            ],
            "electronic": [
                {"pitch": 36, "duration": 0.25},  # Bass drum
                {"pitch": 42, "duration": 0.25},  # Closed hi-hat
                {"pitch": 38, "duration": 0.25},  # Snare
                {"pitch": 42, "duration": 0.25}   # Closed hi-hat
            ]
        }
        
        # Select base pattern
        base_pattern = patterns.get(params.genre, patterns["rock"])
        
        # Modify pattern based on intensity
        if params.intensity > 0.7:
            # Add more frequent hits
            pattern_duration = sum(note["duration"] for note in base_pattern)
            modified_pattern = []
            for note in base_pattern:
                modified_pattern.append(note)
                if note["duration"] >= 0.5:
                    # Add additional hits between longer notes
                    modified_pattern.append({
                        "pitch": note["pitch"],
                        "duration": note["duration"] / 2
                    })
            base_pattern = modified_pattern
        
        # Generate rhythm for full duration
        while current_time < params.duration:
            for note in base_pattern:
                notes.append({
                    "pitch": note["pitch"],
                    "duration": note["duration"],
                    "velocity": random.randint(70, 100) if params.intensity > 0.5 else random.randint(50, 70),
                    "time": current_time
                })
                current_time += note["duration"]
        
        return notes

    def _generate_bass_line(self, scale: List[int], params: MusicParameters) -> List[Dict]:
        """Generate bass line based on scale and genre."""
        notes = []
        current_time = 0
        
        # Get root notes from scale
        root_notes = [scale[0], scale[3], scale[4], scale[5]]  # I, IV, V, VI
        
        while current_time < params.duration:
            # Choose note length based on intensity
            if params.intensity > 0.7:
                duration = random.choice([0.25, 0.5])
                # Add some walking bass patterns
                pattern = [
                    random.choice(root_notes),
                    random.choice(root_notes) + 2,
                    random.choice(root_notes) + 4,
                    random.choice(root_notes) + 5
                ]
                for note_pitch in pattern:
                    notes.append({
                        "pitch": note_pitch - 24,  # Move down 2 octaves
                        "duration": duration,
                        "velocity": random.randint(60, 80),
                        "time": current_time
                    })
                    current_time += duration
            else:
                duration = random.choice([1.0, 2.0])
                notes.append({
                    "pitch": random.choice(root_notes) - 24,
                    "duration": duration,
                    "velocity": random.randint(50, 70),
                    "time": current_time
                })
                current_time += duration
        
        return notes

    def _apply_style(self, notes: List[Dict], style: str, params: MusicParameters) -> List[Dict]:
        """Apply playing style modifications to notes."""
        modified_notes = []
        
        for note in notes:
            if style == "aggressive":
                note["velocity"] = random.randint(90, 127)
                note["duration"] *= 0.8  # Shorter, more aggressive notes
            
            elif style == "gentle":
                note["velocity"] = random.randint(40, 60)
                note["duration"] *= 1.2  # Longer, softer notes
            
            elif style == "staccato":
                note["duration"] *= 0.5
                note["velocity"] = random.randint(70, 90)
            
            elif style == "legato":
                note["duration"] *= 1.1
                # Ensure notes overlap slightly
                if modified_notes:
                    modified_notes[-1]["duration"] += 0.1
            
            elif style == "tremolo":
                # Split note into multiple shorter notes
                sub_duration = note["duration"] / 4
                for i in range(4):
                    modified_notes.append({
                        "pitch": note["pitch"],
                        "duration": sub_duration,
                        "velocity": random.randint(
                            note["velocity"] - 10,
                            note["velocity"] + 10
                        ),
                        "time": note["time"] + (i * sub_duration)
                    })
                continue
            
            modified_notes.append(note)
        
        return modified_notes

class MusicGenerator:
    def __init__(self):
        self.analyzer = PromptAnalyzer()
        self.melody_generator = MelodyGenerator()
    
    def generate_music(self, prompt: str, output_file: str = "output.mid"):
        """Generate music based on text prompt."""
        # Analyze prompt
        params = self.analyzer.analyze_prompt(prompt)
        
        # Generate music for each instrument
        all_notes = []
        for instrument in params.instruments:
            notes = self.melody_generator.generate_for_instrument(instrument, params)
            all_notes.append({
                "instrument": instrument,
                "notes": notes
            })
        
        # Create MIDI file
        self._create_midi(all_notes, params, output_file)
        
        return {
            "status": "success",
            "file": output_file,
            "parameters": vars(params)
        }

    def _create_midi(self, all_notes: List[Dict], params: MusicParameters, filename: str):
        """Create MIDI file from generated notes."""
        midi = midiutil.MIDIFile(len(params.instruments))
        
        # Set tempo
        midi.addTempo(0, 0, params.tempo)
        
        # Add notes for each instrument
        for track, instrument_data in enumerate(all_notes):
            instrument = instrument_data["instrument"]
            notes = instrument_data["notes"]
            
            # Set instrument
            midi.addProgramChange(track, 0, 0, instrument["midi_program"])
            
            # Add notes
            for note in notes:
                midi.addNote(
                    track,
                    0,
                    note["pitch"],
                    note["time"] * params.tempo / 60,
                    note["duration"] * params.tempo / 60,
                    note["velocity"]
                )
        
        # Write file
        with open(filename, "wb") as f:
            midi.writeFile(f)

# Example usage
def main():
    generator = MusicGenerator()
    
    # Example prompts showing different styles and instruments
    prompts = [
        "Create an energetic rock song with heavy electric guitar, driving drums, and a groovy bass line",
        "Compose a peaceful ambient piece with soft synthesizer pads and gentle piano melodies",
        "Generate a jazz tune with walking bass, smooth saxophone solo, and light brush drums",
        "Make an epic orchestral piece with dramatic strings, powerful brass, and thundering timpani",
        "Create a electronic dance track with pulsing synth bass, electronic drums, and atmospheric pads"
    ]
    
    for i, prompt in enumerate(prompts):
        result = generator.generate_music(prompt, f"output_{i}.mid")
        print(f"\nGenerated music for prompt: {prompt}")
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
