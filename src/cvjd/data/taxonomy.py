"""Skill taxonomy loader and robust entity extraction with word boundary guard.

English-focused taxonomy supporting:
1. Exact / alias matching (Layer 1).
2. Taxonomy-based relational matching (Layer 2) with typed relation weights:
   - is_superset_of: 0.80
   - framework_of: 0.60
   - ecosystem_companion: 0.25 - 0.50
   - unrelated: 0.00
3. Asymmetric must-have vs nice-to-have skill classification.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

TAXONOMY_PATH = Path(__file__).resolve().parent / "taxonomy" / "skills_taxonomy.json"
if not TAXONOMY_PATH.exists():
    TAXONOMY_PATH = Path(__file__).resolve().parents[3] / "data" / "taxonomy" / "skills_taxonomy.json"


@dataclass
class SkillEntry:
    id: str
    canonical: str
    category: str
    aliases: List[str] = field(default_factory=list)
    related: List[str] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    custom_regex: Optional[str] = None
    compiled_pattern: Optional[re.Pattern] = None


class SkillTaxonomy:
    """Manages frozen skill taxonomy and provides word-boundary guarded entity extraction."""

    def __init__(self, json_path: Optional[Path] = None) -> None:
        path = json_path or TAXONOMY_PATH
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.version = data.get("version", "1.0.0-frozen")
        self.sha256 = data.get("sha256_checksum", "")
        self.skills_by_id: Dict[str, SkillEntry] = {}
        self.canonical_by_alias: Dict[str, str] = {}
        self.entries_by_canonical: Dict[str, SkillEntry] = {}
        self.entries: List[SkillEntry] = []

        for item in data.get("skills", []):
            entry = SkillEntry(
                id=item["id"],
                canonical=item["canonical"],
                category=item.get("category", "general"),
                aliases=item.get("aliases", []),
                related=item.get("related", []),
                relations=item.get("relations", []),
                custom_regex=item.get("custom_regex"),
            )
            self._compile_entry_pattern(entry)
            self.entries.append(entry)
            self.skills_by_id[entry.id] = entry
            self.entries_by_canonical[entry.canonical.lower()] = entry
            self.canonical_by_alias[entry.canonical.lower()] = entry.canonical

            for alias in entry.aliases:
                self.canonical_by_alias[alias.lower()] = entry.canonical

    def get_entry_by_name(self, name: str) -> Optional[SkillEntry]:
        """Look up SkillEntry by canonical name or alias."""
        clean = unicodedata.normalize("NFC", str(name).strip().lower())
        canonical_name = self.canonical_by_alias.get(clean)
        if canonical_name:
            return self.entries_by_canonical.get(canonical_name.lower())
        return self.entries_by_canonical.get(clean)

    def get_relation_weight(self, source_name: str, target_name: str) -> float:
        """Evaluate Layer 2 (Taxonomy-based Relational Matching) between candidate skill and job requirement.

        Returns:
        - 1.0: exact ID/canonical match
        - 0.8: is_superset_of (e.g. TypeScript -> JavaScript)
        - 0.6: framework_of (e.g. Next.js -> React)
        - 0.25 - 0.50: ecosystem_companion (e.g. Redux -> React)
        - 0.0: No relation (e.g. Next.js != Redux)
        """
        src = self.get_entry_by_name(source_name)
        tgt = self.get_entry_by_name(target_name)
        if not src or not tgt:
            return 0.0
        if src.id == tgt.id:
            return 1.0

        for rel in src.relations:
            if rel.get("target") == tgt.id:
                return float(rel.get("weight", 0.50))
        return 0.0

    def _compile_entry_pattern(self, entry: SkillEntry) -> None:
        if entry.custom_regex:
            entry.compiled_pattern = re.compile(entry.custom_regex, re.IGNORECASE)
            return

        all_variants = [entry.canonical] + entry.aliases
        all_variants = sorted(set(all_variants), key=lambda x: -len(x))
        escaped_variants = [re.escape(v) for v in all_variants if v.strip()]

        joined = "|".join(escaped_variants)
        pattern_str = rf"(?<![a-zA-Z0-9_#+])(?:{joined})(?![a-zA-Z0-9_#+])"
        entry.compiled_pattern = re.compile(pattern_str, re.IGNORECASE)

    def extract_skills_from_text(self, text: str) -> List[str]:
        """Extract canonical skill names from unstructured text."""
        if not text or not text.strip():
            return []

        norm_text = unicodedata.normalize("NFC", text)
        matched_canonicals: List[str] = []
        seen = set()

        for entry in self.entries:
            if entry.compiled_pattern and entry.compiled_pattern.search(norm_text):
                if entry.canonical not in seen:
                    seen.add(entry.canonical)
                    matched_canonicals.append(entry.canonical)

        return matched_canonicals

    def get_related_skills(self, canonical_name: str) -> List[str]:
        """Return related canonical skills."""
        for entry in self.entries:
            if entry.canonical.lower() == canonical_name.lower():
                rel_canonicals = []
                for rel_id in entry.related:
                    if rel_id in self.skills_by_id:
                        rel_canonicals.append(self.skills_by_id[rel_id].canonical)
                return rel_canonicals
        return []

    def classify_jd_skills(self, text: str) -> Tuple[List[str], List[str]]:
        """Classify JD skills into (must_have, nice_to_have)."""
        if not text:
            return [], []

        nice_keywords = [
            "nice to have", "nice-to-have", "preferred", "plus", "bonus", "optional", "advantage"
        ]
        must_keywords = [
            "required", "requirements", "must-have", "must have", "essential", "minimum", "qualifications"
        ]

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        must_lines: List[str] = []
        nice_lines: List[str] = []

        current_mode = "must"
        for line in lines:
            line_lower = line.lower()
            if any(k in line_lower for k in nice_keywords):
                current_mode = "nice"
            elif any(k in line_lower for k in must_keywords):
                current_mode = "must"

            if current_mode == "nice":
                nice_lines.append(line)
            else:
                must_lines.append(line)

        must_skills = self.extract_skills_from_text("\n".join(must_lines))
        nice_skills = self.extract_skills_from_text("\n".join(nice_lines))

        # Prioritize must-have if detected in both
        nice_skills = [s for s in nice_skills if s not in must_skills]

        if not must_skills and not nice_skills:
            must_skills = self.extract_skills_from_text(text)

        return must_skills, nice_skills


_GLOBAL_TAXONOMY: Optional[SkillTaxonomy] = None


def get_taxonomy() -> SkillTaxonomy:
    global _GLOBAL_TAXONOMY
    if _GLOBAL_TAXONOMY is None:
        _GLOBAL_TAXONOMY = SkillTaxonomy()
    return _GLOBAL_TAXONOMY
