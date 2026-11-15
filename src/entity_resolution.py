import re


# Populate this registry using verified authoritative identifiers.
# These entries are illustrative, not a complete biomedical ontology.
ENTITY_REGISTRY = {
    "HGNC:2621": {
        "name": "CYP2C19",
        "entity_type": "GENE",
        "aliases": [
            "CYP2C19",
            "CYP2C19 gene",
            "Cytochrome P450 2C19",
        ],
    },
}


def normalize_entity(name, entity_type="UNKNOWN"):
    name = re.sub(r"\s+", " ", str(name or "").strip())

    if not name:
        return None

    normalized = name.casefold()

    for canonical_id, entity in ENTITY_REGISTRY.items():
        if entity["entity_type"] != entity_type:
            continue

        candidates = [
            entity["name"],
            *entity["aliases"],
        ]

        if normalized in {
            alias.casefold() for alias in candidates
        }:
            return {
                "entity_id": canonical_id,
                "canonical_id": canonical_id,
                "name": entity["name"],
                "entity_type": entity["entity_type"],
                "aliases": entity["aliases"],
            }

    # Unresolved names remain separate. Do not guess identity.
    return {
        "entity_id": f"{entity_type}:{normalized}",
        "canonical_id": "",
        "name": name,
        "entity_type": entity_type,
        "aliases": [],
    }
