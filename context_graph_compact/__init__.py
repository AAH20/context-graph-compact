"""
Context-Graph-Compact: Self-Evolving Lossless Hierarchical DAG Context Compactor & Long-Term Memory Engine.
"""

from .tri_tier_memory import TriTierMemoryEngine, MessageTurn, MemoryTier
from .hebbian import HebbianSemanticGraph, SemanticTriple, SynapticEdge
from .evaluator import SelfEvolutionOracle, EvaluationAuditReport

__version__ = "0.1.0"
__all__ = [
    "TriTierMemoryEngine",
    "MessageTurn",
    "MemoryTier",
    "HebbianSemanticGraph",
    "SemanticTriple",
    "SynapticEdge",
    "SelfEvolutionOracle",
    "EvaluationAuditReport",
]
