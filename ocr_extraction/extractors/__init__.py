"""Extracteurs structurés, un module par type de document."""

from .attestation_extractor import extract_attestation_travail
from .bulletin_extractor import extract_bulletin_salaire
from .cin_extractor import extract_cin
from .quittance_extractor import extract_quittance
from .releve_extractor import extract_releve_bancaire
from .rib_extractor import extract_rib

__all__ = [
    "extract_cin",
    "extract_quittance",
    "extract_rib",
    "extract_releve_bancaire",
    "extract_bulletin_salaire",
    "extract_attestation_travail",
]
