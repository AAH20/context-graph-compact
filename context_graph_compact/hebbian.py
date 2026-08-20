import time
import math
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

@dataclass
class SemanticTriple:
    subject: str
    predicate: str
    object_: str
    confidence: float = 1.0

@dataclass
class SynapticEdge:
    triple: SemanticTriple
    weight: float = 1.0
    access_count: int = 1
    last_activated: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)

class HebbianSemanticGraph:
    """
    Long-Term Semantic Knowledge Graph with Hebbian Synaptic Weighting & Temporal Decay.
    - Neurons that fire together, wire together: successful agent decisions reinforce edge weights.
    - Unused or obsolete historical traces decay exponentially over time.
    """
    def __init__(self, learning_rate: float = 0.15, decay_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.synapses: Dict[str, SynapticEdge] = {} # key: "sub|pred|obj" -> SynapticEdge

    def _key(self, triple: SemanticTriple) -> str:
        return f"{triple.subject.lower()}|{triple.predicate.lower()}|{triple.object_.lower()}"

    def insert_or_reinforce(self, triple: SemanticTriple, reinforcement_boost: float = 1.0) -> SynapticEdge:
        key = self._key(triple)
        now = time.time()

        if key in self.synapses:
            edge = self.synapses[key]
            # Hebbian reinforcement formula: W = W + eta * boost
            edge.weight = round(edge.weight + self.learning_rate * reinforcement_boost, 4)
            edge.access_count += 1
            edge.last_activated = now
            return edge
        else:
            edge = SynapticEdge(
                triple=triple,
                weight=round(1.0 + (self.learning_rate * reinforcement_boost), 4),
                access_count=1,
                last_activated=now,
                created_at=now
            )
            self.synapses[key] = edge
            return edge

    def apply_temporal_decay(self, elapsed_days: float = 1.0):
        """
        Exponential weight decay: W(t) = W(0) * e^(-lambda * t)
        """
        for edge in self.synapses.values():
            decay_factor = math.exp(-self.decay_rate * elapsed_days)
            edge.weight = max(0.01, round(edge.weight * decay_factor, 4))

    def query_facts_for_entity(self, entity: str, min_weight_threshold: float = 0.2) -> List[SemanticTriple]:
        ent_lower = entity.lower()
        results = []
        for edge in sorted(self.synapses.values(), key=lambda e: e.weight, reverse=True):
            if edge.weight >= min_weight_threshold:
                if edge.triple.subject.lower() == ent_lower or edge.triple.object_.lower() == ent_lower:
                    results.append(edge.triple)
        return results

    def export_canonical_invariants(self) -> List[str]:
        lines = []
        for edge in sorted(self.synapses.values(), key=lambda e: e.weight, reverse=True):
            lines.append(f"• ({edge.triple.subject}) --[{edge.triple.predicate}]--> ({edge.triple.object_}) [Synaptic Weight: {edge.weight}]")
        return lines
