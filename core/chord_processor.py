# core/chord_processor.py - Phase 2A: Chord Progression Detection Engine
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class ChordEvent:
    """Represents a single chord event in the timeline"""
    start: float           # Start time in seconds
    end: float            # End time in seconds
    chord: str            # Chord symbol (e.g., "Cm7", "F", "BbMaj7")
    confidence: float     # Detection confidence (0.0 - 1.0)
    quality: str          # major/minor/diminished/augmented/dominant
    root: str             # Root note (C, C#, D, etc.)
    chord_type: str       # triad/7th/9th/11th/13th
    inversion: int        # 0=root, 1=first, 2=second
    chroma_vector: List[float]  # Raw chroma vector for this chord
    downbeat_aligned: bool = False  # Whether this chord aligns with a downbeat

@dataclass
class ChordTimeline:
    """Complete chord progression timeline with metadata"""
    events: List[ChordEvent]
    metadata: Dict[str, Any]
    key_signature: Optional[str] = None
    total_duration: float = 0.0
    resolution: float = 0.1  # Timeline resolution in seconds

class ChordProcessor:
    """
    Main chord detection engine using AudioFlux chroma + template matching
    
    Phase 2A Implementation:
    - 48 basic chord templates (major, minor, 7th, diminished)
    - Template matching with cosine similarity
    - Sub-beat resolution timeline (100ms)
    - Confidence scoring and smoothing
    """
    
    def __init__(self):
        self.chord_templates = {}
        self.chroma_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.min_confidence = 0.6      # Minimum confidence for chord detection
        self.min_chord_duration = 0.5  # Minimum chord duration in seconds
        self.smoothing_window = 3      # Frames for temporal smoothing
        
        self._initialize_chord_templates()
        logger.info("🎵 ChordProcessor initialized with 48 chord templates")
    
    def _initialize_chord_templates(self):
        """Initialize 48 basic chord templates for jazz chord detection"""
        
        # Basic chord intervals (semitones from root)
        chord_intervals = {
            'major': [0, 4, 7],                    # Major triad
            'minor': [0, 3, 7],                    # Minor triad
            'diminished': [0, 3, 6],               # Diminished triad
            'augmented': [0, 4, 8],                # Augmented triad
            'dominant7': [0, 4, 7, 10],            # Dominant 7th
            'major7': [0, 4, 7, 11],               # Major 7th
            'minor7': [0, 3, 7, 10],               # Minor 7th
            'diminished7': [0, 3, 6, 9],           # Diminished 7th
            'half_diminished7': [0, 3, 6, 10],     # Half-diminished 7th (m7b5)
            'minor_major7': [0, 3, 7, 11],         # Minor-major 7th
            'augmented7': [0, 4, 8, 10],           # Augmented 7th
            'suspended2': [0, 2, 7],               # Sus2
            'suspended4': [0, 5, 7],               # Sus4
        }
        
        # Generate all 12 transpositions for each chord type
        for chord_type, intervals in chord_intervals.items():
            for root_idx, root_note in enumerate(self.chroma_names):
                chord_name = self._format_chord_name(root_note, chord_type)
                template = self._create_chord_template(intervals, root_idx)
                self.chord_templates[chord_name] = {
                    'template': template,
                    'root': root_note,
                    'type': chord_type,
                    'intervals': intervals,
                    'quality': self._get_chord_quality(chord_type)
                }
        
        logger.info(f"✅ Generated {len(self.chord_templates)} chord templates")
    
    def _format_chord_name(self, root: str, chord_type: str) -> str:
        """Format chord name according to standard notation"""
        type_symbols = {
            'major': '',
            'minor': 'm',
            'diminished': 'dim',
            'augmented': 'aug',
            'dominant7': '7',
            'major7': 'maj7',
            'minor7': 'm7',
            'diminished7': 'dim7',
            'half_diminished7': 'm7b5',
            'minor_major7': 'mMaj7',
            'augmented7': 'aug7',
            'suspended2': 'sus2',
            'suspended4': 'sus4'
        }
        return f"{root}{type_symbols.get(chord_type, chord_type)}"
    
    def _get_chord_quality(self, chord_type: str) -> str:
        """Get chord quality category"""
        quality_map = {
            'major': 'major',
            'minor': 'minor', 
            'diminished': 'diminished',
            'augmented': 'augmented',
            'dominant7': 'dominant',
            'major7': 'major',
            'minor7': 'minor',
            'diminished7': 'diminished',
            'half_diminished7': 'diminished',
            'minor_major7': 'minor',
            'augmented7': 'augmented',
            'suspended2': 'suspended',
            'suspended4': 'suspended'
        }
        return quality_map.get(chord_type, 'other')
    
    def _create_chord_template(self, intervals: List[int], root_idx: int) -> np.ndarray:
        """Create normalized chroma template for a chord"""
        template = np.zeros(12)
        
        for interval in intervals:
            chroma_idx = (root_idx + interval) % 12
            template[chroma_idx] = 1.0
        
        # Normalize to unit vector
        if np.sum(template) > 0:
            template = template / np.linalg.norm(template)
        
        return template
    
    def analyze_chords(self, chroma_features: Dict[str, Any], downbeats: List[float] = None) -> ChordTimeline:
        """
        Main chord analysis pipeline
        
        Args:
            chroma_features: Output from AudioFlux chroma extraction
            downbeats: Optional downbeat times for musical context
            
        Returns:
            ChordTimeline with detected chord events
        """
        try:
            logger.info("🎵 Starting chord analysis pipeline...")
            
            # Extract chroma data
            chroma_matrix = np.array(chroma_features.get('chroma_matrix', []))
            times = np.array(chroma_features.get('times', []))
            
            if chroma_matrix.size == 0 or times.size == 0:
                logger.error("❌ No chroma data available for chord analysis")
                return ChordTimeline(events=[], metadata={'error': 'No chroma data'})
            
            logger.info(f"🎵 Analyzing {chroma_matrix.shape[1]} chroma frames over {times[-1]:.1f}s")
            
            # Step 1: Frame-level chord detection
            raw_chords = self._detect_chords_frame_level(chroma_matrix, times)
            logger.info(f"🎵 Detected {len(raw_chords)} raw chord frames")
            
            # Step 2: Temporal smoothing and filtering
            smoothed_chords = self._smooth_chord_sequence(raw_chords, times)
            logger.info(f"🎵 After smoothing: {len(smoothed_chords)} chord segments")
            
            # Step 3: Convert to chord events with timing
            chord_events = self._create_chord_events(smoothed_chords, times, downbeats)
            logger.info(f"🎵 Created {len(chord_events)} chord events")
            
            # Step 4: Generate metadata and analysis
            metadata = self._analyze_chord_progression(chord_events, times[-1])
            
            timeline = ChordTimeline(
                events=chord_events,
                metadata=metadata,
                total_duration=times[-1],
                resolution=0.1  # 100ms resolution
            )
            
            logger.info(f"✅ Chord analysis complete: {len(chord_events)} chords detected")
            return timeline
            
        except Exception as e:
            logger.error(f"❌ Chord analysis failed: {e}")
            return ChordTimeline(
                events=[],
                metadata={'error': str(e), 'status': 'failed'}
            )
    
    def _detect_chords_frame_level(self, chroma_matrix: np.ndarray, times: np.ndarray) -> List[Dict[str, Any]]:
        """Detect chords for each chroma frame using template matching"""
        raw_detections = []
        
        for frame_idx in range(chroma_matrix.shape[1]):
            chroma_frame = chroma_matrix[:, frame_idx]
            
            # Normalize frame
            if np.sum(chroma_frame) > 0:
                chroma_frame = chroma_frame / np.linalg.norm(chroma_frame)
            
            # Find best matching chord template
            best_chord, confidence = self._match_chord_template(chroma_frame)
            
            raw_detections.append({
                'time': times[frame_idx],
                'chord': best_chord,
                'confidence': confidence,
                'chroma': chroma_frame.tolist(),
                'frame_idx': frame_idx
            })
        
        return raw_detections
    
    def _match_chord_template(self, chroma_vector: np.ndarray) -> Tuple[str, float]:
        """Find best matching chord template using cosine similarity"""
        best_chord = 'N'  # No chord
        best_score = 0.0
        
        for chord_name, chord_data in self.chord_templates.items():
            template = chord_data['template']
            
            # Cosine similarity
            similarity = np.dot(chroma_vector, template)
            
            if similarity > best_score:
                best_score = similarity
                best_chord = chord_name
        
        # Apply confidence threshold
        if best_score < self.min_confidence:
            best_chord = 'N'
            best_score = 0.0
        
        return best_chord, best_score
    
    def _smooth_chord_sequence(self, raw_chords: List[Dict[str, Any]], times: np.ndarray) -> List[Dict[str, Any]]:
        """Apply temporal smoothing to remove rapid chord changes"""
        if len(raw_chords) < self.smoothing_window:
            return raw_chords
        
        smoothed = []
        
        for i in range(len(raw_chords)):
            # Get window around current frame
            start_idx = max(0, i - self.smoothing_window // 2)
            end_idx = min(len(raw_chords), i + self.smoothing_window // 2 + 1)
            
            window_chords = raw_chords[start_idx:end_idx]
            
            # Find most common chord in window
            chord_counts = {}
            total_confidence = 0
            
            for frame in window_chords:
                chord = frame['chord']
                confidence = frame['confidence']
                
                if chord not in chord_counts:
                    chord_counts[chord] = {'count': 0, 'confidence': 0}
                
                chord_counts[chord]['count'] += 1
                chord_counts[chord]['confidence'] += confidence
                total_confidence += confidence
            
            # Select best chord (highest count * average confidence)
            best_chord = 'N'
            best_score = 0
            
            for chord, data in chord_counts.items():
                avg_confidence = data['confidence'] / data['count']
                score = data['count'] * avg_confidence
                
                if score > best_score:
                    best_score = score
                    best_chord = chord
            
            smoothed.append({
                'time': raw_chords[i]['time'],
                'chord': best_chord,
                'confidence': best_score / len(window_chords),
                'chroma': raw_chords[i]['chroma'],
                'frame_idx': raw_chords[i]['frame_idx']
            })
        
        return smoothed
    
    def _create_chord_events(self, smoothed_chords: List[Dict[str, Any]], times: np.ndarray, 
                           downbeats: List[float] = None) -> List[ChordEvent]:
        """Convert smoothed frame-level chords to chord events with start/end times"""
        if not smoothed_chords:
            return []
        
        events = []
        current_chord = None
        start_time = 0.0
        
        for i, frame in enumerate(smoothed_chords):
            chord = frame['chord']
            time = frame['time']
            confidence = frame['confidence']
            
            # Check if chord changed
            if chord != current_chord:
                # End previous chord event
                if current_chord is not None and current_chord != 'N':
                    end_time = time
                    duration = end_time - start_time
                    
                    # Only add chord if it meets minimum duration
                    if duration >= self.min_chord_duration:
                        chord_data = self.chord_templates.get(current_chord, {})
                        
                        events.append(ChordEvent(
                            start=start_time,
                            end=end_time,
                            chord=current_chord,
                            confidence=prev_confidence,
                            quality=chord_data.get('quality', 'unknown'),
                            root=chord_data.get('root', 'N'),
                            chord_type=self._classify_chord_type(chord_data.get('type', '')),
                            inversion=0,  # TODO: Implement inversion detection
                            chroma_vector=prev_chroma,
                            downbeat_aligned=self._is_downbeat_aligned(start_time, downbeats)
                        ))
                
                # Start new chord
                current_chord = chord
                start_time = time
                prev_confidence = confidence
                prev_chroma = frame['chroma']
        
        # Add final chord
        if current_chord is not None and current_chord != 'N':
            end_time = times[-1]
            duration = end_time - start_time
            
            if duration >= self.min_chord_duration:
                chord_data = self.chord_templates.get(current_chord, {})
                
                events.append(ChordEvent(
                    start=start_time,
                    end=end_time,
                    chord=current_chord,
                    confidence=prev_confidence,
                    quality=chord_data.get('quality', 'unknown'),
                    root=chord_data.get('root', 'N'),
                    chord_type=self._classify_chord_type(chord_data.get('type', '')),
                    inversion=0,
                    chroma_vector=prev_chroma,
                    downbeat_aligned=self._is_downbeat_aligned(start_time, downbeats)
                ))
        
        return events
    
    def _classify_chord_type(self, chord_type: str) -> str:
        """Classify chord type into categories"""
        if '7' in chord_type:
            return '7th'
        elif chord_type in ['major', 'minor']:
            return 'triad'
        else:
            return 'other'
    
    def _is_downbeat_aligned(self, chord_time: float, downbeats: List[float], tolerance: float = 0.2) -> bool:
        """Check if chord timing aligns with a downbeat"""
        if not downbeats:
            return False
        
        for downbeat in downbeats:
            if abs(chord_time - downbeat) <= tolerance:
                return True
        
        return False
    
    def _analyze_chord_progression(self, events: List[ChordEvent], duration: float) -> Dict[str, Any]:
        """Analyze chord progression for musical patterns and metadata"""
        if not events:
            return {'status': 'no_chords', 'total_chords': 0}
        
        # Basic statistics
        chord_names = [event.chord for event in events]
        unique_chords = list(set(chord_names))
        
        # Chord quality distribution
        qualities = [event.quality for event in events]
        quality_counts = {q: qualities.count(q) for q in set(qualities)}
        
        # Average chord duration
        durations = [event.end - event.start for event in events]
        avg_duration = np.mean(durations)
        
        # Downbeat alignment percentage
        aligned_count = sum(1 for event in events if event.downbeat_aligned)
        alignment_percentage = (aligned_count / len(events)) * 100 if events else 0
        
        metadata = {
            'status': 'success',
            'total_chords': len(events),
            'unique_chords': len(unique_chords),
            'chord_vocabulary': unique_chords,
            'quality_distribution': quality_counts,
            'average_chord_duration': avg_duration,
            'downbeat_alignment_percentage': alignment_percentage,
            'timeline_coverage': sum(durations) / duration * 100,
            'processing_method': 'audioflux_template_matching',
            'template_count': len(self.chord_templates)
        }
        
        return metadata
    
    def get_chord_color(self, chord_event: ChordEvent) -> str:
        """Get color for chord visualization based on quality and root"""
        
        # Color scheme for different chord qualities
        color_schemes = {
            'major': {
                'C': '#3B82F6', 'D': '#1E40AF', 'E': '#172554', 'F': '#155E75',
                'G': '#1D4ED8', 'A': '#1E3A8A', 'B': '#0F172A',
                'C#': '#075985', 'D#': '#0369A1', 'F#': '#0284C7', 
                'G#': '#0891B2', 'A#': '#0E7490'
            },
            'minor': {
                'C': '#EF4444', 'D': '#B91C1C', 'E': '#7F1D1D', 'F': '#CA8A04',
                'G': '#DC2626', 'A': '#991B1B', 'B': '#450A0A',
                'C#': '#B45309', 'D#': '#D97706', 'F#': '#F59E0B',
                'G#': '#F59E0B', 'A#': '#EAB308'
            },
            'dominant': {
                'C': '#8B5CF6', 'D': '#6D28D9', 'E': '#4C1D95', 'F': '#7C2D12',
                'G': '#7C3AED', 'A': '#5B21B6', 'B': '#3C1677',
                'C#': '#581C87', 'D#': '#6B21A8', 'F#': '#86198F',
                'G#': '#A21CAF', 'A#': '#BE185D'
            },
            'diminished': {
                'C': '#6B7280', 'D': '#4B5563', 'E': '#374151', 'F': '#1F2937',
                'G': '#9CA3AF', 'A': '#6B7280', 'B': '#4B5563',
                'C#': '#374151', 'D#': '#1F2937', 'F#': '#111827',
                'G#': '#6B7280', 'A#': '#4B5563'
            }
        }
        
        quality = chord_event.quality
        root = chord_event.root
        
        return color_schemes.get(quality, color_schemes['major']).get(root, '#6B7280')