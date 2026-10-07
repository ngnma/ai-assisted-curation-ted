"""Convert the reviewed thesaurus CSV into SKOS, the standard format for thesauri (used by ELSST and Annif).

Input:  vocab/thesaurus.csv     (tag, broader), reviewed by a human
Output: vocab/ted_thesaurus.ttl
Run from the project root:  python -m src.export_skos
"""

from urllib.parse import quote

import pandas as pd
from rdflib import Graph, Literal, Namespace
from rdflib.namespace import RDF, SKOS

from src.config import VOCAB_DIR

BASE = Namespace("https://example.org/ted-thesaurus/")  # placeholder web address that identifies each concept


def concept_uri(kind: str, label: str):
    return BASE[f"{kind}/{quote(label.replace(' ', '_'))}"]


def main() -> None:
    df = pd.read_csv(VOCAB_DIR / "thesaurus.csv")
    assert df["tag"].is_unique, "each tag must appear exactly once"
    assert not (df["broader"] == "UNASSIGNED").any(), "assign every tag before exporting"

    g = Graph()
    g.bind("skos", SKOS)
    scheme = BASE["scheme"]
    g.add((scheme, RDF.type, SKOS.ConceptScheme))
    g.add((scheme, SKOS.prefLabel, Literal("TED mini-thesaurus", lang="en")))

    for broader in df["broader"].unique():  # top-level concepts
        b = concept_uri("group", broader)
        g.add((b, RDF.type, SKOS.Concept))
        g.add((b, SKOS.prefLabel, Literal(broader, lang="en")))
        g.add((b, SKOS.topConceptOf, scheme))
        g.add((b, SKOS.inScheme, scheme))

    for tag, broader in zip(df["tag"], df["broader"]):  # tags, linked to their parent
        t, b = concept_uri("tag", tag), concept_uri("group", broader)
        g.add((t, RDF.type, SKOS.Concept))
        g.add((t, SKOS.prefLabel, Literal(tag, lang="en")))
        g.add((t, SKOS.inScheme, scheme))
        g.add((t, SKOS.broader, b))
        g.add((b, SKOS.narrower, t))

    out = VOCAB_DIR / "ted_thesaurus.ttl"
    g.serialize(out, format="turtle")
    print(f"Saved {len(df)} tags under {df['broader'].nunique()} broader concepts to {out}")


if __name__ == "__main__":
    main()