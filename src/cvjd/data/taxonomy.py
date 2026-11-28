"""Skill taxonomy loader and robust entity extraction with word boundary guard.

English-focused taxonomy supporting:
1. Exact / alias matching (Layer 1) with bucketed regex pre-filtering.
2. Taxonomy-based relational matching (Layer 2) with 1-hop & decayed 2-hop
   traversal over typed edges (`is_superset_of`, `framework_of`, `specialized_in`,
   `ecosystem_companion`).
3. Optional Layer 3 semantic matching via cached skill embeddings.
4. Clause/heading-aware asymmetric must-have vs nice-to-have JD classification
   with negation guarding and dynamic boundary renormalization.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

TAXONOMY_PATH = Path(__file__).resolve().parent / "taxonomy" / "skills_taxonomy.json"
if not TAXONOMY_PATH.exists():
    TAXONOMY_PATH = Path(__file__).resolve().parents[3] / "data" / "taxonomy" / "skills_taxonomy.json"

_HEADING_NICE_RE = re.compile(
    r"^(?:nice\s*[- ]to\s*[- ]have|preferred(?:\s+(?:skills|qualifications|experience))?|"
    r"bonus(?:\s+(?:skills|points))?|desired(?:\s+skills)?|optional(?:\s+skills)?|"
    r"plus(?:es)?|good\s+to\s+have|advantages?)\s*:?\s*$",
    re.IGNORECASE,
)
_HEADING_MUST_RE = re.compile(
    r"^(?:required(?:\s+(?:skills|qualifications|experience))?|requirements|must\s*[- ]have|"
    r"essential(?:\s+(?:skills|qualifications))?|minimum\s+qualifications|"
    r"qualifications|core\s+skills|technical\s+skills|skills\s+required)\s*:?\s*$",
    re.IGNORECASE,
)
_CLAUSE_NICE_CUES = (
    "nice to have",
    "nice-to-have",
    "good to have",
    "is a plus",
    "are a plus",
    "as a plus",
    "would be a plus",
    "be a plus",
    "preferred",
    "bonus",
    "optional",
    "an advantage",
    "added advantage",
    "desirable",
)
_CLAUSE_MUST_CUES = (
    "required",
    "requirements",
    "must-have",
    "must have",
    "essential",
    "mandatory",
    "minimum",
    "qualifications",
)
_NEGATED_CLAUSE_RE = re.compile(
    r"\b(?:no\s+(?:prior|previous)\s+[^.;,\n]{1,45}?\s+(?:required|needed|necessary)|"
    r"not\s+(?:required|needed|necessary)\s*:\s*[^.;\n]+|"
    r"without\s+(?:requiring|needing)\s+[^.;,\n]+)",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\s*;\s*")


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
        self._adj: Dict[str, List[Tuple[str, float]]] = {}
        self._prefilter_checks: List[Tuple[Optional[Tuple[str, ...]], SkillEntry]] = []

        # Optional Layer 3 semantic state
        self._encode_fn: Optional[Callable[[List[str]], np.ndarray]] = None
        self._skill_id_to_emb_idx: Dict[str, int] = {}
        self._skill_embeddings: Optional[np.ndarray] = None
        self._oov_emb_cache: Dict[str, np.ndarray] = {}

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

        self._build_relation_graph()
        self._build_fast_prefilter()

    def _build_relation_graph(self) -> None:
        """Precompute directed weighted adjacency list from typed relations."""
        self._adj.clear()
        for entry in self.entries:
            edges: List[Tuple[str, float]] = []
            for rel in entry.relations:
                target_id = rel.get("target")
                if target_id and target_id in self.skills_by_id:
                    edges.append((target_id, float(rel.get("weight", 0.50))))
            self._adj[entry.id] = edges

    def _build_fast_prefilter(self) -> None:
        """Precompute lowercase substring pre-checks for non-custom_regex entries."""
        self._prefilter_checks.clear()
        for entry in self.entries:
            if entry.custom_regex:
                self._prefilter_checks.append((None, entry))
            else:
                variants = tuple(
                    sorted(
                        {
                            unicodedata.normalize("NFC", v).lower()
                            for v in ([entry.canonical] + entry.aliases)
                            if v.strip()
                        },
                        key=len,
                    )
                )
                self._prefilter_checks.append((variants, entry))

    def get_entry_by_name(self, name: str) -> Optional[SkillEntry]:
        """Look up SkillEntry by canonical name or alias."""
        clean = unicodedata.normalize("NFC", str(name).strip().lower())
        canonical_name = self.canonical_by_alias.get(clean)
        if canonical_name:
            return self.entries_by_canonical.get(canonical_name.lower())
        return self.entries_by_canonical.get(clean)

    def get_relation_weight(
        self,
        source_name: str,
        target_name: str,
        max_hops: int = 2,
        hop_decay: float = 0.85,
    ) -> float:
        """Evaluate Layer 2 (Taxonomy-based Relational Matching) between candidate skill and job requirement.

        Supports direct 1-hop typed relations and decayed multi-hop transitive
        chains over typed edges (e.g. NestJS -> TypeScript -> JavaScript).
        """
        src = self.get_entry_by_name(source_name)
        tgt = self.get_entry_by_name(target_name)
        if not src or not tgt:
            return 0.0
        if src.id == tgt.id:
            return 1.0

        # 1-hop direct relation takes precedence
        best_direct = 0.0
        for nxt_id, w in self._adj.get(src.id, ()):
            if nxt_id == tgt.id and w > best_direct:
                best_direct = w
        if best_direct > 0.0 or max_hops <= 1:
            return best_direct

        # Bounded BFS for transitive chains (max_hops >= 2)
        best_transitive = 0.0
        queue: deque[Tuple[str, float, int]] = deque([(src.id, 1.0, 0)])
        best_seen: Dict[str, float] = {src.id: 1.0}

        while queue:
            curr_id, curr_w, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for nxt_id, edge_w in self._adj.get(curr_id, ()):
                next_depth = depth + 1
                decayed_w = curr_w * edge_w * (hop_decay if next_depth > 1 else 1.0)
                if nxt_id == tgt.id:
                    if decayed_w > best_transitive:
                        best_transitive = decayed_w
                    continue
                if decayed_w > best_seen.get(nxt_id, 0.0):
                    best_seen[nxt_id] = decayed_w
                    queue.append((nxt_id, decayed_w, next_depth))

        return float(min(best_transitive, 0.75))

    def attach_skill_embeddings(
        self,
        encode_fn: Callable[[List[str]], np.ndarray],
        include_category: bool = True,
    ) -> None:
        """Pre-encode and cache canonical skill vectors for optional Layer 3 matching."""
        texts = [
            f"{e.canonical} ({e.category.replace('_', ' ')})" if include_category else e.canonical
            for e in self.entries
        ]
        vecs = np.asarray(encode_fn(texts), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms > 1e-9, norms, 1.0)
        self._skill_embeddings = vecs / norms
        self._skill_id_to_emb_idx = {e.id: idx for idx, e in enumerate(self.entries)}
        self._encode_fn = encode_fn
        self._oov_emb_cache.clear()

    def _get_skill_vector(self, name: str) -> Optional[np.ndarray]:
        """Return L2-normalized embedding for a canonical/alias or OOV skill phrase."""
        if self._skill_embeddings is None:
            return None
        entry = self.get_entry_by_name(name)
        if entry is not None:
            idx = self._skill_id_to_emb_idx.get(entry.id)
            if idx is not None:
                return self._skill_embeddings[idx]
        if self._encode_fn is None:
            return None
        key = unicodedata.normalize("NFC", str(name).strip().lower())
        if not key:
            return None
        cached = self._oov_emb_cache.get(key)
        if cached is not None:
            return cached
        vec = np.asarray(self._encode_fn([name]), dtype=np.float32)[0]
        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec = vec / norm
        self._oov_emb_cache[key] = vec
        return vec

    def get_semantic_relation_weight(
        self,
        source_name: str,
        target_name: str,
        min_sim: float = 0.72,
        max_credit: float = 0.45,
    ) -> float:
        """Evaluate Layer 3 continuous semantic similarity between two skill names."""
        v_src = self._get_skill_vector(source_name)
        v_tgt = self._get_skill_vector(target_name)
        if v_src is None or v_tgt is None:
            return 0.0
        sim = float(np.dot(v_src, v_tgt))
        if sim < min_sim:
            return 0.0
        # Map [min_sim, 1.0] linearly into [0.15, max_credit]
        ratio = (sim - min_sim) / max(1.0 - min_sim, 1e-6)
        return float(min(max(0.15 + ratio * (max_credit - 0.15), 0.0), max_credit))

    def get_effective_relation_weight(
        self,
        source_name: str,
        target_name: str,
        use_layer2: bool = True,
        use_layer3: bool = True,
        max_hops: int = 2,
    ) -> float:
        """Combined Layer 1 -> Layer 2 -> Layer 3 relation weight in [0, 1]."""
        src = self.get_entry_by_name(source_name)
        tgt = self.get_entry_by_name(target_name)
        if src is not None and tgt is not None and src.id == tgt.id:
            return 1.0
        if not src and not tgt:
            clean_src = unicodedata.normalize("NFC", str(source_name).strip().lower())
            clean_tgt = unicodedata.normalize("NFC", str(target_name).strip().lower())
            if clean_src and clean_src == clean_tgt:
                return 1.0

        if use_layer2:
            rel_w = self.get_relation_weight(source_name, target_name, max_hops=max_hops)
            if rel_w > 0.0:
                return rel_w

        if use_layer3 and self._skill_embeddings is not None:
            return self.get_semantic_relation_weight(source_name, target_name)

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

    def extract_skills_from_text(self, text: str, strip_negated: bool = False) -> List[str]:
        """Extract canonical skill names from unstructured text."""
        if not text or not text.strip():
            return []

        norm_text = unicodedata.normalize("NFC", text)
        if strip_negated:
            norm_text = _NEGATED_CLAUSE_RE.sub(" ", norm_text)
        lower_text = norm_text.lower()

        matched_canonicals: List[str] = []
        seen: Set[str] = set()

        for variants, entry in self._prefilter_checks:
            if variants is not None and not any(v in lower_text for v in variants):
                continue
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

    @staticmethod
    def _is_heading_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if _HEADING_NICE_RE.match(stripped) or _HEADING_MUST_RE.match(stripped):
            return True
        words = stripped.split()
        return len(words) <= 6 and stripped.endswith(":")

    def classify_jd_skills(self, text: str) -> Tuple[List[str], List[str]]:
        """Classify JD skills into (must_have, nice_to_have) using heading + clause-level cues."""
        if not text:
            return [], []

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        must_clauses: List[str] = []
        nice_clauses: List[str] = []

        section_mode = "must"
        for line in lines:
            line_clean = _NEGATED_CLAUSE_RE.sub(" ", line)
            line_lower = line_clean.lower()

            if self._is_heading_line(line_clean):
                if any(k in line_lower for k in _CLAUSE_NICE_CUES):
                    section_mode = "nice"
                elif any(k in line_lower for k in _CLAUSE_MUST_CUES):
                    section_mode = "must"
                continue

            # Evaluate sentence/clause level so inline "is a plus" does not flip the whole JD
            clauses = [c.strip() for c in _SENTENCE_SPLIT_RE.split(line_clean) if c.strip()]
            for clause in clauses:
                clause_lower = clause.lower()
                has_nice = any(k in clause_lower for k in _CLAUSE_NICE_CUES)
                has_must = any(k in clause_lower for k in _CLAUSE_MUST_CUES)

                if has_nice and not has_must:
                    nice_clauses.append(clause)
                elif has_must and not has_nice:
                    must_clauses.append(clause)
                elif has_must and has_nice:
                    # Mixed sentence: default to must unless explicitly framed as bonus/optional
                    if any(k in clause_lower for k in ("nice to have", "nice-to-have", "bonus", "optional")):
                        nice_clauses.append(clause)
                    else:
                        must_clauses.append(clause)
                else:
                    if section_mode == "nice":
                        nice_clauses.append(clause)
                    else:
                        must_clauses.append(clause)

        must_skills = self.extract_skills_from_text("\n".join(must_clauses), strip_negated=True)
        nice_skills = self.extract_skills_from_text("\n".join(nice_clauses), strip_negated=True)

        # Prioritize must-have if detected in both
        must_set = set(must_skills)
        nice_skills = [s for s in nice_skills if s not in must_set]

        if not must_skills and not nice_skills:
            must_skills = self.extract_skills_from_text(text, strip_negated=True)

        return must_skills, nice_skills

    def compute_skill_idf(self, corpus_texts: Iterable[str]) -> Dict[str, float]:
        """Compute smooth normalized IDF weights for canonical skills over a text corpus."""
        docs = list(corpus_texts)
        n_docs = len(docs)
        if n_docs == 0:
            return {}

        df: Dict[str, int] = {}
        for doc in docs:
            for skill in self.extract_skills_from_text(doc):
                df[skill] = df.get(skill, 0) + 1

        if not df:
            return {}

        raw_idf = {
            skill: 1.0 + math.log((n_docs + 1.0) / (count + 1.0))
            for skill, count in df.items()
        }
        mean_idf = sum(raw_idf.values()) / len(raw_idf)
        if mean_idf <= 0.0:
            return {skill: 1.0 for skill in raw_idf}
        return {skill: val / mean_idf for skill, val in raw_idf.items()}

    def score_skills_coverage(
        self,
        must_skills: Sequence[str],
        nice_skills: Sequence[str],
        cv_skills: Sequence[str],
        skill_weights: Optional[Dict[str, float]] = None,
        must_weight: float = 0.85,
        nice_weight: float = 0.15,
        neutral_score: float = 1.0,
        use_layer2: bool = True,
        use_layer3: bool = True,
        max_hops: int = 2,
    ) -> float:
        """Unified asymmetric skill coverage with Layer 2/3 & dynamic boundary renormalization."""
        if not must_skills and not nice_skills:
            return float(neutral_score)

        cv_set = set(cv_skills)

        def _cov(targets: Sequence[str]) -> float:
            if not targets:
                return 0.0
            total_w = 0.0
            covered_w = 0.0
            for tgt in targets:
                w = float(skill_weights.get(tgt, 1.0)) if skill_weights else 1.0
                if w <= 0.0:
                    continue
                total_w += w
                if tgt in cv_set:
                    best = 1.0
                else:
                    best = 0.0
                    for cs in cv_skills:
                        rel = self.get_effective_relation_weight(
                            cs,
                            tgt,
                            use_layer2=use_layer2,
                            use_layer3=use_layer3,
                            max_hops=max_hops,
                        )
                        if rel > best:
                            best = rel
                            if best >= 1.0:
                                break
                covered_w += w * best
            return min(covered_w / total_w, 1.0) if total_w > 0.0 else 0.0

        if must_skills and not nice_skills:
            return float(_cov(must_skills))
        if nice_skills and not must_skills:
            return float(_cov(nice_skills))

        must_cov = _cov(must_skills)
        nice_cov = _cov(nice_skills)
        denom = must_weight + nice_weight
        if denom <= 0.0:
            return float(neutral_score)
        return float((must_weight * must_cov + nice_weight * nice_cov) / denom)


_GLOBAL_TAXONOMY: Optional[SkillTaxonomy] = None


def get_taxonomy() -> SkillTaxonomy:
    global _GLOBAL_TAXONOMY
    if _GLOBAL_TAXONOMY is None:
        _GLOBAL_TAXONOMY = SkillTaxonomy()
    return _GLOBAL_TAXONOMY
