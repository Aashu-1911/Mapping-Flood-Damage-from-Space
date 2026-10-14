"""Spaceborne Flood Damage Mapping & Multi-Modal Disaster Intelligence (Track B).

Multimodal AI Hackathon 2026:
- Event: August 26, 2026 Glacial Collapse & Debris Flood in Bhote Koshi - Trishuli Corridor (Nepal).
- Sentinel-1 SAR & Sentinel-2 Optical Change Detection.
- OSM Infrastructure Damage Assessment.
- Cut-Off Settlement Graph Analysis.
- AI Flood Segmentation & Grounded Bilingual Copilot.
"""

from .contracts import (
    DamageSummary,
    CutoffSettlement,
    ValleyTraceResult,
    ValidationMetrics,
    SituationReportResponse,
    CopilotQueryRequest,
    CopilotQueryResponse,
)

__all__ = [
    "DamageSummary",
    "CutoffSettlement",
    "ValleyTraceResult",
    "ValidationMetrics",
    "SituationReportResponse",
    "CopilotQueryRequest",
    "CopilotQueryResponse",
]
