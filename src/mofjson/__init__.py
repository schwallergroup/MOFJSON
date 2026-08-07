"""MOFJSON — a token-efficient, semantically rich JSON representation of MOFs for LLMs.

Phase 01.1. This module is the *only* thing users should need to import from.
"""

from mofjson.schema.metal_node import MetalNode, MetalSite
from mofjson.schema.molecular import MolFragment
from mofjson.version import __version__

__all__ = ["MetalNode", "MetalSite", "MolFragment", "__version__"]
