#!/usr/bin/env python3
"""
Module 1: GenAI-Based P&A Data Extraction
==========================================
Implements ideas from:
  Kolay et al., "From Archives to Abandonment: Applying Generative AI
  to Optimize Plug and Abandon Processes in Old Oil Wells,"
  Petrophysics, vol. 66, no. 4, pp. 545–554, August 2025.

Key concepts:
  - OCR-based text extraction from scanned well reports
  - Semantic chunking of extracted text
  - Vector embedding and similarity search (RAG pipeline)
  - Rule-based extraction of hole/casing/cement data
  - Quality-control checks (e.g., casing OD < hole size)
"""

import re
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Domain data structures
# ---------------------------------------------------------------------------
@dataclass
class CasingRecord:
    """A single casing string record extracted from a well report."""
    name: str
    od_inches: float            # outer diameter
    id_inches: float            # inner diameter
    weight_ppf: float           # weight lb/ft
    grade: str
    top_depth_ft: float
    bottom_depth_ft: float
    toc_depth_ft: Optional[float] = None  # top of cement


@dataclass
class HoleSection:
    """Open-hole section extracted from a well report."""
    name: str
    diameter_inches: float
    top_depth_ft: float
    bottom_depth_ft: float


@dataclass
class WellSchematic:
    """Aggregated well schematic built from extracted data."""
    well_name: str = ""
    hole_sections: List[HoleSection] = field(default_factory=list)
    casings: List[CasingRecord] = field(default_factory=list)
    qc_issues: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. Simulated OCR  (replaces real Tesseract / cloud OCR)
# ---------------------------------------------------------------------------
def simulate_ocr(raw_text: str, error_rate: float = 0.02) -> str:
    """Simulate OCR noise on a text string.

    Parameters
    ----------
    raw_text : str
        Ground-truth text (the 'scanned' document).
    error_rate : float
        Probability of character-level OCR error.

    Returns
    -------
    str
        Text with simulated OCR artefacts.
    """
    confusions = {
        'O': '0', '0': 'O', 'l': '1', '1': 'l',
        'I': 'l', 'S': '5', '5': 'S', 'B': '8',
    }
    chars = list(raw_text)
    rng = random.Random(42)
    for i, ch in enumerate(chars):
        if rng.random() < error_rate and ch in confusions:
            chars[i] = confusions[ch]
    return "".join(chars)


# ---------------------------------------------------------------------------
# 2. Text chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for embedding.

    Parameters
    ----------
    text : str
        Input document text.
    chunk_size : int
        Approximate number of characters per chunk.
    overlap : int
        Character overlap between consecutive chunks.

    Returns
    -------
    list of str
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# 3. Lightweight vector store (cosine similarity on bag-of-words TF vectors)
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class SimpleVectorStore:
    """Minimal TF-based vector store for semantic search."""

    def __init__(self):
        self.chunks: List[str] = []
        self.vectors: List[Dict[str, float]] = []

    def add(self, chunk: str) -> None:
        tokens = _tokenize(chunk)
        tf: Dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        norm = math.sqrt(sum(v * v for v in tf.values())) or 1.0
        tf = {k: v / norm for k, v in tf.items()}
        self.chunks.append(chunk)
        self.vectors.append(tf)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[float, str]]:
        qtokens = _tokenize(query)
        qtf: Dict[str, float] = {}
        for t in qtokens:
            qtf[t] = qtf.get(t, 0) + 1
        qnorm = math.sqrt(sum(v * v for v in qtf.values())) or 1.0
        qtf = {k: v / qnorm for k, v in qtf.items()}

        scores = []
        for idx, vec in enumerate(self.vectors):
            dot = sum(qtf.get(k, 0) * v for k, v in vec.items())
            scores.append((dot, self.chunks[idx]))
        scores.sort(key=lambda x: -x[0])
        return scores[:top_k]


# ---------------------------------------------------------------------------
# 4. Rule-based data extraction (simulates LLM prompt-engineered extraction)
# ---------------------------------------------------------------------------
_CASING_PATTERN = re.compile(
    r"(?P<name>\w[\w\s]*?casing|tubing|liner)"
    r"[:\s]+"
    r"(?P<od>[\d.]+)\s*in\.?\s*OD"
    r"[,;\s]+"
    r"(?P<wt>[\d.]+)\s*(?:lb/ft|ppf)"
    r"[,;\s]+"
    r"(?P<grade>[A-Z]\-?\d+)"
    r"[,;\s]+set\s+(?:at|from)\s+(?P<top>[\d,]+)\s*(?:to\s+(?P<bot>[\d,]+))?\s*ft",
    re.IGNORECASE,
)

_HOLE_PATTERN = re.compile(
    r"(?P<name>\w[\w\s]*?hole)"
    r"[:\s]+"
    r"(?P<dia>[\d.]+)\s*in\.?"
    r"[,;\s]+(?:from\s+)?(?P<top>[\d,]+)\s*to\s+(?P<bot>[\d,]+)\s*ft",
    re.IGNORECASE,
)

_TOC_PATTERN = re.compile(
    r"top\s+of\s+cement[:\s]+(?P<toc>[\d,]+)\s*ft",
    re.IGNORECASE,
)


def extract_well_data(text: str) -> WellSchematic:
    """Extract hole, casing, and TOC data from text via regex rules.

    This mirrors the LLM + prompt-engineering approach described
    by Kolay et al., replacing the LLM with deterministic regex rules.

    Parameters
    ----------
    text : str
        Concatenated relevant text chunks.

    Returns
    -------
    WellSchematic
    """
    schema = WellSchematic()

    for m in _HOLE_PATTERN.finditer(text):
        schema.hole_sections.append(HoleSection(
            name=m.group("name").strip(),
            diameter_inches=float(m.group("dia")),
            top_depth_ft=float(m.group("top").replace(",", "")),
            bottom_depth_ft=float(m.group("bot").replace(",", "")),
        ))

    for m in _CASING_PATTERN.finditer(text):
        bot_str = m.group("bot")
        bot = float(bot_str.replace(",", "")) if bot_str else float(m.group("top").replace(",", ""))
        od = float(m.group("od"))
        casing = CasingRecord(
            name=m.group("name").strip(),
            od_inches=od,
            id_inches=round(od - 0.5, 3),   # simplified
            weight_ppf=float(m.group("wt")),
            grade=m.group("grade"),
            top_depth_ft=float(m.group("top").replace(",", "")),
            bottom_depth_ft=bot,
        )
        schema.casings.append(casing)

    # Search for TOC
    toc_match = _TOC_PATTERN.search(text)
    if toc_match and schema.casings:
        schema.casings[-1].toc_depth_ft = float(
            toc_match.group("toc").replace(",", "")
        )

    return schema


# ---------------------------------------------------------------------------
# 5. Quality-control checks
# ---------------------------------------------------------------------------
def run_qc_checks(schema: WellSchematic) -> List[str]:
    """Run QC checks analogous to those in the Kolay et al. interactive tool.

    Checks include:
      - Casing OD must be smaller than hole diameter at the same depth.
      - Casing strings must nest properly (inner OD < outer ID).
      - TOC must be above the casing shoe.
    """
    issues: List[str] = []

    for csg in schema.casings:
        for hs in schema.hole_sections:
            if (csg.top_depth_ft >= hs.top_depth_ft and
                    csg.bottom_depth_ft <= hs.bottom_depth_ft):
                if csg.od_inches >= hs.diameter_inches:
                    issues.append(
                        f"QC FAIL: {csg.name} OD ({csg.od_inches} in.) "
                        f">= hole size ({hs.diameter_inches} in.)")

    sorted_csg = sorted(schema.casings, key=lambda c: c.od_inches)
    for i in range(1, len(sorted_csg)):
        inner, outer = sorted_csg[i - 1], sorted_csg[i]
        if inner.od_inches >= outer.id_inches:
            issues.append(
                f"QC FAIL: {inner.name} OD ({inner.od_inches}) "
                f"does not fit inside {outer.name} ID ({outer.id_inches})")

    for csg in schema.casings:
        if csg.toc_depth_ft is not None and csg.toc_depth_ft > csg.bottom_depth_ft:
            issues.append(
                f"QC FAIL: TOC ({csg.toc_depth_ft} ft) below shoe "
                f"({csg.bottom_depth_ft} ft) for {csg.name}")

    schema.qc_issues = issues
    return issues


# ---------------------------------------------------------------------------
# 6. End-to-end pipeline
# ---------------------------------------------------------------------------
def pa_extraction_pipeline(document_text: str, query: str = "casing hole cement") -> WellSchematic:
    """Full RAG-style extraction pipeline.

    1. Simulate OCR
    2. Chunk text
    3. Build vector store and retrieve relevant chunks
    4. Extract structured data
    5. Run QC
    """
    ocr_text = simulate_ocr(document_text, error_rate=0.01)
    chunks = chunk_text(ocr_text, chunk_size=400, overlap=80)
    store = SimpleVectorStore()
    for ch in chunks:
        store.add(ch)
    results = store.search(query, top_k=5)
    relevant_text = "\n".join(txt for _, txt in results)
    schema = extract_well_data(relevant_text)
    run_qc_checks(schema)
    return schema


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_all():
    """Test the PA GenAI extraction pipeline with synthetic well report data."""
    report = (
        "WELL REPORT - WELL A-42\n"
        "Surface hole: 26.0 in., from 0 to 500 ft\n"
        "Intermediate hole: 17.5 in., from 500 to 3000 ft\n"
        "Production hole: 12.25 in., from 3000 to 8500 ft\n\n"
        "Surface casing: 20.0 in. OD, 94.0 lb/ft, K-55, set from 0 to 480 ft\n"
        "Intermediate casing: 13.375 in. OD, 68.0 lb/ft, N-80, set from 0 to 2900 ft\n"
        "Production casing: 9.625 in. OD, 47.0 lb/ft, P-110, set from 0 to 8400 ft\n"
        "Top of cement: 2200 ft\n"
    )

    # 1. OCR
    ocr = simulate_ocr(report)
    assert isinstance(ocr, str) and len(ocr) == len(report)

    # 2. Chunking
    chunks = chunk_text(report, 200, 40)
    assert len(chunks) >= 2

    # 3. Vector store
    store = SimpleVectorStore()
    for ch in chunks:
        store.add(ch)
    hits = store.search("casing diameter", top_k=2)
    assert len(hits) == 2

    # 4. Extraction
    schema = extract_well_data(report)
    assert len(schema.hole_sections) == 3, f"Expected 3 holes, got {len(schema.hole_sections)}"
    assert len(schema.casings) == 3, f"Expected 3 casings, got {len(schema.casings)}"
    assert schema.casings[-1].toc_depth_ft == 2200.0

    # 5. QC
    issues = run_qc_checks(schema)
    assert len(issues) == 0, f"Unexpected QC issues: {issues}"

    # 6. Trigger a deliberate QC failure
    schema.casings[0].od_inches = 30.0
    issues = run_qc_checks(schema)
    assert any("QC FAIL" in i for i in issues)

    # 7. Full pipeline
    schema2 = pa_extraction_pipeline(report)
    assert isinstance(schema2, WellSchematic)

    print("[PASS] pa_genai_extraction — all tests passed")


if __name__ == "__main__":
    test_all()
