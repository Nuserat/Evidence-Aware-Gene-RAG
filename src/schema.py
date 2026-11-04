
from typing import Literal
from pydantic import BaseModel, Field


EntityType = Literal[
    "GENE",
    "VARIANT",
    "DRUG",
    "CHEMICAL",
    "PROTEIN",
    "DISEASE",
    "PHENOTYPE",
    "PATHWAY",
    "UNKNOWN",
]

EvidencePolarity = Literal[
    "POSITIVE",
    "NEGATIVE",
    "UNCERTAIN",
    "NOT_ASSESSED",
]


class Entity(BaseModel):
    entity_id: str
    name: str
    entity_type: EntityType = "UNKNOWN"

    # Authoritative identifiers, e.g. HGNC:2621, rs4244285
    canonical_id: str = ""

    # Alternative names found in source databases
    aliases: list[str] = Field(default_factory=list)


class EvidenceProvenance(BaseModel):
    evidence_id: str

    source: str = ""
    source_id: str = ""

    pmid: str = ""
    doi: str = ""
    url: str = ""

    evidence_text: str = ""
    evidence_type: str = "literature"

    dataset_version: str = ""
    publication_year: int | None = None


class RelationshipAssertion(BaseModel):
    assertion_id: str

    subject: Entity
    predicate: str
    object: Entity

    provenance: EvidenceProvenance

    # Must be supplied by an annotation or validated classifier.
    evidence_polarity: EvidencePolarity = "NOT_ASSESSED"

    # Optional source-provided or calibrated confidence.
    confidence: float | None = None

    # Preserve source-specific metadata.
    statistical_significance: str = ""
    effect_direction: str = ""
    effect_size: str = ""

    original_record: dict = Field(default_factory=dict)
