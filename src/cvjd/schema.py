# src/cvjd/schema.py

from dataclasses import dataclass, field
from typing import Literal

SCHEMA_VERSION = "1.0.0"

Lang = Literal["vi", "en", "mixed"]
DocType = Literal["cv", "jd"]


@dataclass(frozen=True)
class Section:
    """
    A normalized section of a CV or JD.

    Attributes:
        name: Normalized section name, e.g. "skills", "experience".
        text: Text content of the section.
        heading_raw: Original heading, e.g. "Kỹ năng" or "TECHNICAL SKILLS".
        order: Position of the section in the original document.
    """

    name: str
    text: str
    heading_raw: str = ""
    order: int = 0


@dataclass(frozen=True)
class Document:
    """
    Represents a CV or Job Description document.

    Attributes:
        doc_id: Unique document identifier.
        doc_type: "cv" or "jd".
        lang: Document language.
        raw_text: Original document text.
        sections: Mapping from normalized section name to Section.
        schema_version: Version of the data schema.
    """

    doc_id: str
    doc_type: DocType
    lang: Lang
    raw_text: str
    sections: dict[str, Section] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class Pair:
    """
    Represents a CV-JD matching pair.

    Attributes:
        cv_id: CV document ID.
        jd_id: Job Description document ID.
        label: Relevance label:
            - 0: not relevant
            - 1: weakly relevant
            - 2: relevant
            - 3: highly relevant
            - -1: not labeled yet
    """

    cv_id: str
    jd_id: str
    label: int


@dataclass(frozen=True)
class Query:
    """
    Represents a ranking query.

    Attributes:
        jd_id: Job Description used as the query.
        cv_ids: Candidate CV IDs.
        relevance: Relevance labels corresponding to cv_ids.
    """

    jd_id: str
    cv_ids: list[str]
    relevance: list[int]
