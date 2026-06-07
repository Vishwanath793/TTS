"""
Text & PDF-to-Audio Pipeline v3.1 — Multi-Voice Cinematic Edition
==========================================================
Pipeline: pdfplumber (extract + font analysis) / txt → Role detection 
          → Gemini (polish) → edge-tts (multi-voice) → MP3 stitching

NEW IN v3.1:
  • Added native support for plain .txt files alongside PDFs.
"""

import os
import re
import json
import hashlib
import argparse
import asyncio
import time
import tempfile
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter
from pathlib import Path
from typing import Optional

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from google import genai
except ImportError:
    genai = None

import edge_tts
from functools import lru_cache


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEXT SEGMENT ROLES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Role:
    HEADER    = "HEADER"      # Chapter / section titles
    BODY      = "BODY"        # Normal paragraph text
    BOLD      = "BOLD"        # Bold / emphasized text
    ITALIC    = "ITALIC"      # Italic / aside text
    QUOTE     = "QUOTE"       # Blockquotes or quoted speech
    AUTHOR    = "AUTHOR"      # Attribution: "-- Name" or "— Name"
    LIST_ITEM = "LIST_ITEM"   # Bullet / numbered list items
    PAUSE     = "PAUSE"       # Silent pause between segments


@dataclass
class Segment:
    """A chunk of text with an assigned voice role."""
    role: str
    text: str
    page: int = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VOICE PROFILES — each role maps to (voice, rate, volume)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VOICE_PROFILES = {
    # --- "default" profile: clear distinction between roles ---
    "default": {
        Role.HEADER:    {"voice": "en-US-GuyNeural",          "rate": "-10%", "volume": "+0%"},
        Role.BODY:      {"voice": "en-US-AriaNeural",         "rate": "+0%",  "volume": "+0%"},
        Role.BOLD:      {"voice": "en-US-AriaNeural",         "rate": "-8%",  "volume": "+0%"},
        Role.ITALIC:    {"voice": "en-US-JennyNeural",        "rate": "+3%",  "volume": "+0%"},
        Role.QUOTE:     {"voice": "en-US-GuyNeural",          "rate": "-12%", "volume": "+0%"},
        Role.AUTHOR:    {"voice": "en-US-JennyNeural",        "rate": "-5%",  "volume": "+0%"},
        Role.LIST_ITEM: {"voice": "en-US-AriaNeural",         "rate": "+0%",  "volume": "+0%"},
    },

    # --- "audiobook" — warm male narrator + female for asides ---
    "audiobook": {
        Role.HEADER:    {"voice": "en-US-ChristopherNeural",  "rate": "-15%", "volume": "+0%"},
        Role.BODY:      {"voice": "en-US-ChristopherNeural",  "rate": "-5%",  "volume": "+0%"},
        Role.BOLD:      {"voice": "en-US-ChristopherNeural",  "rate": "-10%", "volume": "+0%"},
        Role.ITALIC:    {"voice": "en-US-MichelleNeural",     "rate": "+0%",  "volume": "+0%"},
        Role.QUOTE:     {"voice": "en-US-EricNeural",         "rate": "-8%",  "volume": "+0%"},
        Role.AUTHOR:    {"voice": "en-US-MichelleNeural",     "rate": "-5%",  "volume": "+0%"},
        Role.LIST_ITEM: {"voice": "en-US-ChristopherNeural",  "rate": "-3%",  "volume": "+0%"},
    },

    # --- "cinematic" — dramatic contrast between voices ---
    "cinematic": {
        Role.HEADER:    {"voice": "en-US-RogerNeural",        "rate": "-20%", "volume": "+0%"},
        Role.BODY:      {"voice": "en-US-EmmaNeural",         "rate": "+0%",  "volume": "+0%"},
        Role.BOLD:      {"voice": "en-US-EmmaNeural",         "rate": "-10%", "volume": "+0%"},
        Role.ITALIC:    {"voice": "en-US-MichelleNeural",     "rate": "+5%",  "volume": "+0%"},
        Role.QUOTE:     {"voice": "en-US-RogerNeural",        "rate": "-10%", "volume": "+0%"},
        Role.AUTHOR:    {"voice": "en-US-MichelleNeural",     "rate": "-8%",  "volume": "+0%"},
        Role.LIST_ITEM: {"voice": "en-US-EmmaNeural",         "rate": "+0%",  "volume": "+0%"},
    },

    # --- "indian-english" — Indian English voices ---
    "indian": {
        Role.HEADER:    {"voice": "en-IN-PrabhatNeural",      "rate": "-10%", "volume": "+0%"},
        Role.BODY:      {"voice": "en-IN-NeerjaNeural",       "rate": "+0%",  "volume": "+0%"},
        Role.BOLD:      {"voice": "en-IN-NeerjaNeural",       "rate": "-8%",  "volume": "+0%"},
        Role.ITALIC:    {"voice": "en-IN-NeerjaNeural",       "rate": "+5%",  "volume": "+0%"},
        Role.QUOTE:     {"voice": "en-IN-PrabhatNeural",      "rate": "-10%", "volume": "+0%"},
        Role.AUTHOR:    {"voice": "en-IN-NeerjaNeural",       "rate": "-5%",  "volume": "+0%"},
        Role.LIST_ITEM: {"voice": "en-IN-NeerjaNeural",       "rate": "+0%",  "volume": "+0%"},
    },

    # --- "minimal" — single voice, only rate changes ---
    "minimal": {
        Role.HEADER:    {"voice": "en-US-AriaNeural",         "rate": "-15%", "volume": "+0%"},
        Role.BODY:      {"voice": "en-US-AriaNeural",         "rate": "+0%",  "volume": "+0%"},
        Role.BOLD:      {"voice": "en-US-AriaNeural",         "rate": "-5%",  "volume": "+0%"},
        Role.ITALIC:    {"voice": "en-US-AriaNeural",         "rate": "+5%",  "volume": "+0%"},
        Role.QUOTE:     {"voice": "en-US-AriaNeural",         "rate": "-10%", "volume": "+0%"},
        Role.AUTHOR:    {"voice": "en-US-AriaNeural",         "rate": "-5%",  "volume": "+0%"},
        Role.LIST_ITEM: {"voice": "en-US-AriaNeural",         "rate": "+0%",  "volume": "+0%"},
    },
}

# Pause durations — dot count controls filler length (kept very short)
PAUSE_BETWEEN = {
    (Role.HEADER, Role.BODY):       "...",        # brief pause after headers
    (Role.BODY, Role.HEADER):       "...",        # brief pause before new header
    (Role.BODY, Role.QUOTE):        "..",         # tiny pause into quote
    (Role.QUOTE, Role.BODY):        "..",         # tiny pause out of quote
    (Role.QUOTE, Role.AUTHOR):      "",           # no gap before attribution
    (Role.BODY, Role.LIST_ITEM):    "..",
    (Role.LIST_ITEM, Role.LIST_ITEM): "",
    (Role.LIST_ITEM, Role.BODY):    "..",
    "default":                       "..",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_MODEL      = "gemini-3-flash-preview"
DEFAULT_CHUNK_SIZE = 5
CACHE_DIR          = ".tts_cache"
MAX_RETRIES        = 1
RETRY_BASE_DELAY   = 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEMINI PROMPT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TTS_OPTIMIZER_PROMPT = """You are an Audio Script Editor optimizing extracted text for Text-to-Speech.

RULES:
1. PRESERVE original meaning and wording. Do NOT summarize or add commentary.
2. FIX extraction artifacts: re-join hyphenated breaks, merge split sentences, remove surviving headers/footers.
3. ADD spoken pauses ("...") after titles, between topic shifts, before lists.
4. Replace em-dashes with commas. Convert parenthetical asides to commas.
5. Expand remaining abbreviations, spell out symbols (& → and, @ → at, # → number).
6. Convert any surviving tabular data to spoken prose.
7. Remove footnote markers, citation numbers, URLs.
8. For code snippets: describe purpose in one sentence, don't read syntax.

CRITICAL: The text uses role markers like [HEADER], [BOLD], [ITALIC], [QUOTE], [AUTHOR], [LIST].
You MUST preserve these markers exactly as they appear. Only clean the text between markers.

OUTPUT: Only the final script with markers preserved. No markdown, no code fences, no commentary."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1: FONT-AWARE EXTRACTION (PDF ONLY)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class FontRun:
    """A contiguous run of characters with the same font properties."""
    text: str
    fontname: str
    size: float
    is_bold: bool
    is_italic: bool
    top: float    # y-position on page
    page: int
    page_height: float = 0.0


def extract_font_runs(pdf_path: str) -> list[FontRun]:
    """
    Extract text as font-annotated runs from every page.
    Each run = consecutive characters sharing the same font properties.
    """
    if pdfplumber is None:
        return []

    all_runs = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            chars = page.chars
            if not chars:
                continue

            pg_height = float(page.height) if page.height else 792.0

            # Group chars into runs by font properties
            current_font = None
            current_size = None
            current_text = ""
            current_top = 0

            for ch in chars:
                fname = ch.get("fontname", "")
                fsize = round(ch.get("size", 0), 1)
                text = ch.get("text", "")
                top = ch.get("top", 0)

                if fname == current_font and fsize == current_size:
                    # Check if we jumped to a new line (large y-gap)
                    if abs(top - current_top) > fsize * 1.8 and current_text.strip():
                        # New line — flush current run and start fresh
                        all_runs.append(FontRun(
                            text=current_text,
                            fontname=current_font or "",
                            size=current_size or 0,
                            is_bold=_is_bold(current_font or ""),
                            is_italic=_is_italic(current_font or ""),
                            top=current_top,
                            page=page_idx + 1,
                            page_height=pg_height,
                        ))
                        current_text = text
                        current_top = top
                    else:
                        current_text += text
                else:
                    # Font changed — flush
                    if current_text.strip():
                        all_runs.append(FontRun(
                            text=current_text,
                            fontname=current_font or "",
                            size=current_size or 0,
                            is_bold=_is_bold(current_font or ""),
                            is_italic=_is_italic(current_font or ""),
                            top=current_top,
                            page=page_idx + 1,
                            page_height=pg_height,
                        ))
                    current_font = fname
                    current_size = fsize
                    current_text = text
                    current_top = top

            # Flush last run
            if current_text.strip():
                all_runs.append(FontRun(
                    text=current_text,
                    fontname=current_font or "",
                    size=current_size or 0,
                    is_bold=_is_bold(current_font or ""),
                    is_italic=_is_italic(current_font or ""),
                    top=current_top,
                    page=page_idx + 1,
                    page_height=pg_height,
                ))

    return all_runs


@lru_cache(maxsize=256)
def _is_bold(fontname: str) -> bool:
    fn = fontname.lower()
    return any(k in fn for k in ("bold", "heavy", "black", "demi", "semibold"))


@lru_cache(maxsize=256)
def _is_italic(fontname: str) -> bool:
    fn = fontname.lower()
    return any(k in fn for k in ("italic", "oblique", "slant"))


def _detect_body_size(runs: list[FontRun]) -> float:
    """Find the most common font size — that's the body text."""
    if not runs:
        return 12.0
    sizes = Counter(r.size for r in runs if r.text.strip())
    return sizes.most_common(1)[0][0]


# Page number / footer patterns
_FOOTER_PAGE_RE = re.compile(
    r'(page\s+\d+|\bpg\.?\s*\d+|\d+\s*$|·\s*page\s*\d+|\|\s*page\s*\d+)',
    re.IGNORECASE,
)
_RE_DIGITS = re.compile(r'\d+')

def _remove_margin_runs(runs: list[FontRun], total_pages: int) -> list[FontRun]:
    """
    Remove font runs that sit in the top/bottom 8% of the page and repeat
    (with numbers normalized) across >30% of pages.  This catches running
    headers/footers *before* they get merged with body text.
    """
    if total_pages < 3:
        return runs

    MARGIN_FRAC = 0.08  # top/bottom 8 % of the page

    # Collect margin-zone runs, normalized
    margin_texts: dict[str, set[int]] = {}   # normalized_text → set of pages
    for r in runs:
        if r.page_height <= 0:
            continue
        in_top = r.top < r.page_height * MARGIN_FRAC
        in_bot = r.top > r.page_height * (1 - MARGIN_FRAC)
        if not (in_top or in_bot):
            continue
        norm = _RE_DIGITS.sub('#', r.text.strip())
        if not norm:
            continue
        margin_texts.setdefault(norm, set()).add(r.page)

    threshold = total_pages * 0.30
    repeated = {norm for norm, pages in margin_texts.items() if len(pages) >= threshold}

    if not repeated:
        return runs

    filtered = []
    removed = 0
    for r in runs:
        norm = _RE_DIGITS.sub('#', r.text.strip())
        in_top = r.page_height > 0 and r.top < r.page_height * MARGIN_FRAC
        in_bot = r.page_height > 0 and r.top > r.page_height * (1 - MARGIN_FRAC)
        if (in_top or in_bot) and norm in repeated:
            removed += 1
            continue
        filtered.append(r)

    if removed:
        print(f"  🧹 Removed {removed} header/footer runs from page margins.")
    return filtered


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2: CONVERT FONT RUNS → ROLE-TAGGED SEGMENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Patterns for role detection beyond fonts
AUTHOR_PATTERN  = re.compile(r'^[\s]*[-–—]\s*(.+)$')           # "— John Doe" or "- Author"
QUOTE_PATTERN   = re.compile(r'^["\u201c\u201d\u2018\u2019]')  # starts with quote mark
LIST_PATTERN    = re.compile(r'^\s*(?:[-•*✓→▶]|\d+[\.\)])\s')  # bullet or numbered

HEADING_PATTERNS = [
    re.compile(r'^(Chapter|CHAPTER)\s+\d+', re.IGNORECASE),
    re.compile(r'^(Section|SECTION)\s+\d+', re.IGNORECASE),
    re.compile(r'^(Part|PART)\s+(One|Two|Three|Four|Five|\d+|[IVXivx]+)', re.IGNORECASE),
    re.compile(r'^\d+\.\s+[A-Z]'),
    re.compile(r'^(Introduction|Conclusion|Abstract|Summary|References|Appendix)\s*$', re.IGNORECASE),
]


def font_runs_to_segments(runs: list[FontRun]) -> list[Segment]:
    """
    Convert font-annotated runs into role-tagged segments.
    Uses font size/style + regex patterns to assign roles.
    """
    if not runs:
        return []

    body_size = _detect_body_size(runs)
    segments = []

    for run in runs:
        text = run.text.strip()
        if not text:
            continue

        role = _classify_run(run, text, body_size)
        segments.append(Segment(role=role, text=text, page=run.page))

    # Post-process: merge adjacent segments of the same role
    segments = _merge_adjacent(segments)

    # Post-process: apply text-pattern detection on BODY segments
    segments = _apply_pattern_roles(segments)

    return segments


def _classify_run(run: FontRun, text: str, body_size: float) -> str:
    """Classify a single font run by font properties."""
    size_ratio = run.size / body_size if body_size > 0 else 1.0

    # Significantly larger font → HEADER
    if size_ratio >= 1.25:
        return Role.HEADER

    # Bold text at body size → BOLD emphasis
    if run.is_bold and not run.is_italic:
        # But if it's very short and starts a line, it might be a sub-header
        if len(text.split()) <= 6 and text[0].isupper():
            return Role.HEADER
        return Role.BOLD

    # Italic → ITALIC aside
    if run.is_italic:
        return Role.ITALIC

    # Non-bold, non-italic text that is slightly larger than body (ratio 1.10–1.24)
    # → subtitle / sub-heading.  Use ITALIC role so it gets a distinct voice and
    # never merges with the main HEADER or the following BODY paragraph.
    if size_ratio >= 1.10:
        return Role.ITALIC

    return Role.BODY


def _merge_adjacent(segments: list[Segment]) -> list[Segment]:
    """Merge consecutive segments of the same role into one."""
    if not segments:
        return []

    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg.role == prev.role and seg.page == prev.page:
            prev.text += " " + seg.text
        else:
            merged.append(seg)
    return merged


def _apply_pattern_roles(segments: list[Segment]) -> list[Segment]:
    """Apply regex-based role detection on top of font-based classification."""
    result = []
    for seg in segments:
        text = seg.text.strip()

        # Author attribution: "-- Name" or "— Name" at end of a quote-like context
        am = AUTHOR_PATTERN.match(text)
        if am and len(text) < 80:
            name = am.group(1).strip()
            result.append(Segment(Role.AUTHOR, f"by {name}", seg.page))
            continue

        # Quoted text (starts with quotation mark)
        if QUOTE_PATTERN.match(text) and seg.role == Role.BODY:
            seg.role = Role.QUOTE
            result.append(seg)
            continue

        # ALL-CAPS short text → heading
        if text.isupper() and len(text.split()) >= 2 and len(text) < 80 and seg.role == Role.BODY:
            seg.role = Role.HEADER
            result.append(seg)
            continue

        # Heading patterns
        is_heading = False
        for pat in HEADING_PATTERNS:
            if pat.match(text):
                is_heading = True
                break
        if is_heading and seg.role == Role.BODY:
            seg.role = Role.HEADER
            result.append(seg)
            continue

        # List items
        if LIST_PATTERN.match(text) and seg.role == Role.BODY:
            seg.role = Role.LIST_ITEM
            result.append(seg)
            continue

        result.append(seg)

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2b: FALLBACK / TXT FILE — TEXT-ONLY SEGMENTATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def text_to_segments(text: str, page: int = 1) -> list[Segment]:
    """
    When pdfplumber is unavailable, font extraction fails, OR when using a .txt file.
    Segments text purely by regex patterns.
    """
    lines = text.split('\n')
    segments = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Author attribution
        am = AUTHOR_PATTERN.match(stripped)
        if am and len(stripped) < 80:
            segments.append(Segment(Role.AUTHOR, f"by {am.group(1).strip()}", page))
            continue

        # ALL-CAPS heading
        if stripped.isupper() and len(stripped.split()) >= 2 and len(stripped) < 80:
            segments.append(Segment(Role.HEADER, stripped, page))
            continue

        # Chapter/section heading
        is_heading = False
        for pat in HEADING_PATTERNS:
            if pat.match(stripped):
                is_heading = True
                break
        if is_heading:
            segments.append(Segment(Role.HEADER, stripped, page))
            continue

        # Quoted text
        if QUOTE_PATTERN.match(stripped):
            segments.append(Segment(Role.QUOTE, stripped, page))
            continue

        # List items
        if LIST_PATTERN.match(stripped):
            segments.append(Segment(Role.LIST_ITEM, stripped, page))
            continue

        # Default body
        segments.append(Segment(Role.BODY, stripped, page))

    return _merge_adjacent(segments)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2c: REPEATED HEADER/FOOTER REMOVAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def remove_repeated_segments(segments: list[Segment], total_pages: int) -> list[Segment]:
    """Remove text that appears on >40% of pages (headers/footers)."""
    if total_pages < 4:
        return segments

    # Count distinct pages each normalized text appears on
    norm_pages: dict[str, set[int]] = {}
    for seg in segments:
        norm = _RE_DIGITS.sub('#', seg.text.strip())
        if norm:
            norm_pages.setdefault(norm, set()).add(seg.page)

    threshold = total_pages * 0.4
    repeated = {norm for norm, pages in norm_pages.items() if len(pages) >= threshold}

    if not repeated:
        return segments

    filtered = []
    removed = 0
    for seg in segments:
        if _RE_DIGITS.sub('#', seg.text.strip()) in repeated:
            removed += 1
        else:
            filtered.append(seg)

    if removed:
        print(f"  🧹 Removed {removed} repeated header/footer segments.")

    return filtered


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3: LOCAL TEXT CLEANUP (same NLP as v2, applied per segment)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ABBREVIATIONS = {
    r'\be\.g\.': 'for example,', r'\bi\.e\.': 'that is,', r'\betc\.': 'and so on',
    r'\bvs\.': 'versus', r'\bDr\.': 'Doctor', r'\bMr\.': 'Mister',
    r'\bMrs\.': 'Missus', r'\bMs\.': 'Ms', r'\bProf\.': 'Professor',
    r'\bFig\.': 'Figure', r'\bfig\.': 'figure', r'\bNo\.': 'Number',
    r'\bVol\.': 'Volume', r'\bSec\.': 'Section', r'\bApprox\.': 'Approximately',
    r'\bgovt\.': 'government', r'\bdept\.': 'department', r'\bw/o\b': 'without',
}

UNITS = {
    r'(\d)\s*km\b': r'\1 kilometers', r'(\d)\s*kg\b': r'\1 kilograms',
    r'(\d)\s*mg\b': r'\1 milligrams', r'(\d)\s*ml\b': r'\1 milliliters',
    r'(\d)\s*cm\b': r'\1 centimeters', r'(\d)\s*mm\b': r'\1 millimeters',
    r'(\d)\s*%': r'\1 percent', r'(\d)\s*°C\b': r'\1 degrees Celsius',
    r'(\d)\s*°F\b': r'\1 degrees Fahrenheit',
    r'(\d)\s*GB\b': r'\1 gigabytes', r'(\d)\s*MB\b': r'\1 megabytes',
    r'(\d)\s*TB\b': r'\1 terabytes', r'(\d)\s*fps\b': r'\1 frames per second',
    r'(\d)\s*mph\b': r'\1 miles per hour',
}

# Pre-compiled substitutions for clean_segment_text (avoids re-compiling on every call)
_ABBREV_COMPILED = [(re.compile(p), r) for p, r in ABBREVIATIONS.items()]
_UNITS_COMPILED  = [(re.compile(p), r) for p, r in UNITS.items()]

_RE_HYPHEN_BREAK  = re.compile(r'(\w)-\s*\n\s*(\w)')
_RE_CID           = re.compile(r'\(cid:\d+\)')
_RE_CITE_NUM      = re.compile(r'\s*\[\d+(?:[,\-–]\s*\d+)*\]')
_RE_CITE_AUTH     = re.compile(r'\s*\([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][a-z]+)*,?\s*\d{4}[a-z]?\)')
_RE_DAGGERS       = re.compile(r'[†‡§¶]')
_RE_URL_HTTP      = re.compile(r'https?://\S+')
_RE_URL_WWW       = re.compile(r'www\.\S+')
_RE_PAGE_NUM      = re.compile(r'^\s*\d{1,4}\s*$', re.MULTILINE)
_RE_AMPERSAND     = re.compile(r'\s*&\s*')
_RE_AT            = re.compile(r'@')
_RE_HASH_NUM      = re.compile(r'(?<!\w)#(\d+)')
_RE_TILDE_NUM     = re.compile(r'~(\d)')
_RE_GTE           = re.compile(r'≥')
_RE_LTE           = re.compile(r'≤')
_RE_ARROW         = re.compile(r'→')
_RE_CURRENCY      = re.compile(r'([$£€₹])\s*(\d[\d,]*\.?\d*)\s*([KkMmBbTt])?(?=\s|$|[,.\)])')
_RE_ISO_DATE      = re.compile(r'\b(\d{4})[-/](\d{2})[-/](\d{2})\b')
_RE_EM_DASH       = re.compile(r'\s*[—–]\s*')
_RE_SPACES        = re.compile(r'[ \t]+')
_RE_NEWLINES      = re.compile(r'\n{2,}')

_ORDINALS = {
    '1st': 'first', '2nd': 'second', '3rd': 'third', '4th': 'fourth',
    '5th': 'fifth', '10th': 'tenth', '20th': 'twentieth', '21st': 'twenty-first',
}
_ORDINALS_COMPILED = [(re.compile(rf'\b{num}\b'), word) for num, word in _ORDINALS.items()]

_MONTHS = {
    '01': 'January', '02': 'February', '03': 'March', '04': 'April',
    '05': 'May', '06': 'June', '07': 'July', '08': 'August',
    '09': 'September', '10': 'October', '11': 'November', '12': 'December',
}
_CURRENCY_MAP = {"$": "dollars", "£": "pounds", "€": "euros", "₹": "rupees"}
_MULTIPLIER_MAP = {"K": "thousand", "M": "million", "B": "billion", "T": "trillion"}


def clean_segment_text(text: str) -> str:
    """Apply all local NLP cleanup to a segment's text."""
    text = _RE_HYPHEN_BREAK.sub(r'\1\2', text)
    text = _RE_CID.sub('', text)
    text = _RE_CITE_NUM.sub('', text)
    text = _RE_CITE_AUTH.sub('', text)
    text = _RE_DAGGERS.sub('', text)
    text = _RE_URL_HTTP.sub('', text)
    text = _RE_URL_WWW.sub('', text)
    text = _RE_PAGE_NUM.sub('', text)

    for pat, rep in _ABBREV_COMPILED:
        text = pat.sub(rep, text)
    for pat, rep in _UNITS_COMPILED:
        text = pat.sub(rep, text)

    text = _RE_AMPERSAND.sub(' and ', text)
    text = _RE_AT.sub(' at ', text)
    text = _RE_HASH_NUM.sub(r'number \1', text)
    text = _RE_TILDE_NUM.sub(r'approximately \1', text)
    text = _RE_GTE.sub('greater than or equal to ', text)
    text = _RE_LTE.sub('less than or equal to ', text)
    text = _RE_ARROW.sub('leads to ', text)

    def _money(m):
        sym, amt, suf = m.group(1), m.group(2), (m.group(3) or "").upper()
        cur = _CURRENCY_MAP.get(sym, "dollars")
        mul = _MULTIPLIER_MAP.get(suf, "")
        return f"{amt} {mul} {cur}".strip() if mul else f"{amt} {cur}"
    text = _RE_CURRENCY.sub(_money, text)

    for pat, word in _ORDINALS_COMPILED:
        text = pat.sub(word, text)

    def _iso(m):
        return f"{_MONTHS.get(m.group(2), m.group(2))} {int(m.group(3))}, {m.group(1)}"
    text = _RE_ISO_DATE.sub(_iso, text)

    text = _RE_EM_DASH.sub(', ', text)
    text = _RE_SPACES.sub(' ', text)
    text = _RE_NEWLINES.sub(' ', text)

    return text.strip()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4: GEMINI AI OPTIMIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _call_gemini(client, model: str, prompt: str, system: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model, contents=prompt,
                config={"system_instruction": system, "max_output_tokens": 8192, "temperature": 0.15},
            )
            return response.text.strip()
        except Exception as e:
            if any(k in str(e).lower() for k in ("rate", "429", "quota")):
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"    ⏳ Rate limited. Retry in {delay}s...")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"Gemini failed after {MAX_RETRIES} retries.")


def segments_to_marked_text(segments: list[Segment]) -> str:
    """Convert segments into marker-delimited text for Gemini."""
    parts = []
    for seg in segments:
        parts.append(f"[{seg.role}] {seg.text}")
    return "\n\n".join(parts)


def marked_text_to_segments(text: str, original_segments: list[Segment]) -> list[Segment]:
    """Parse Gemini output back into segments. Preserves role markers."""
    pattern = re.compile(r'\[(HEADER|BODY|BOLD|ITALIC|QUOTE|AUTHOR|LIST_ITEM)\]\s*(.*?)(?=\n\s*\[(?:HEADER|BODY|BOLD|ITALIC|QUOTE|AUTHOR|LIST_ITEM)\]|\Z)', re.DOTALL)
    matches = pattern.findall(text)

    if not matches:
        # Gemini didn't preserve markers — return original segments with cleaned text
        return original_segments

    return [Segment(role=role, text=body.strip()) for role, body in matches if body.strip()]


def ai_optimize_segments(segments: list[Segment], client, model: str) -> list[Segment]:
    """Send role-marked text to Gemini and get back optimized segments."""
    marked = segments_to_marked_text(segments)

    prompt = (
        "Here is role-tagged extracted text. Each section starts with a role marker like [HEADER], [BODY], etc.\n"
        "Optimize the text for TTS while preserving ALL role markers exactly as they are.\n"
        "Only clean/improve the text between markers.\n\n"
        f"{marked}"
    )

    result = _call_gemini(client, model, prompt, TTS_OPTIMIZER_PROMPT)
    return marked_text_to_segments(result, segments)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 5: MULTI-VOICE AUDIO GENERATION + STITCHING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def generate_segment_audio(text: str, voice: str, rate: str, volume: str, path: str):
    """Generate a single MP3 segment."""
    communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume)
    await communicate.save(path)


async def generate_pause_audio(pause_text: str, voice: str, path: str):
    """Generate a very short near-silent pause MP3."""
    if not pause_text:
        return False
    # edge-tts 7.x rejects pure punctuation.  Speak a single short word
    # at maximum speed and minimum volume to create a brief gap.
    try:
        communicate = edge_tts.Communicate("a", voice, rate="+50%", volume="-100%")
        await communicate.save(path)
        return True
    except Exception:
        return False


async def stitch_segments_to_mp3(
    segments: list[Segment],
    profile: dict,
    output_path: str,
    write_srt: bool = False,
):
    """
    Generate audio for each segment with its assigned voice,
    then concatenate all MP3 chunks into one file.
    """
    tmp_dir = tempfile.mkdtemp(prefix="tts_segments_")
    chunk_paths = []
    srt_entries = []
    time_offset_ms = 0

    _RE_ALPHA_NUM = re.compile(r'[^a-zA-Z0-9]')

    async def _render_one(idx: int, seg: Segment):
        if seg.role == Role.PAUSE:
            return None
        if not _RE_ALPHA_NUM.sub('', seg.text):
            return None
        voice_cfg = profile.get(seg.role, profile[Role.BODY])
        chunk_path = os.path.join(tmp_dir, f"{idx:05d}_segment.mp3")
        try:
            await generate_segment_audio(
                seg.text, voice_cfg["voice"], voice_cfg["rate"], voice_cfg["volume"], chunk_path
            )
            return chunk_path
        except Exception:
            return None

    try:
        # Generate all segments concurrently
        results = await asyncio.gather(*(_render_one(i, s) for i, s in enumerate(segments)))
        chunk_paths = [p for p in results if p is not None]

        # Concatenate all MP3 chunks (MP3 is frame-based, binary concat works)
        with open(output_path, "wb") as out_f:
            for cp in chunk_paths:
                if os.path.exists(cp) and os.path.getsize(cp) > 0:
                    with open(cp, "rb") as in_f:
                        out_f.write(in_f.read())

        # Generate SRT if requested (single-voice fallback for timing)
        if write_srt:
            await _generate_srt_for_segments(segments, profile, output_path)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def _generate_srt_for_segments(segments: list[Segment], profile: dict, mp3_path: str):
    """Generate an approximate SRT file based on segment text."""
    srt_path = mp3_path.rsplit('.', 1)[0] + ".srt"
    srt_lines = []
    # Estimate: ~150 words per minute, adjust for rate
    idx = 1
    time_ms = 0

    for seg in segments:
        if seg.role == Role.PAUSE:
            time_ms += 500
            continue

        words = len(seg.text.split())
        voice_cfg = profile.get(seg.role, profile[Role.BODY])

        # Parse rate to adjust WPM
        rate_str = voice_cfg.get("rate", "+0%")
        rate_pct = int(re.search(r'([+-]?\d+)', rate_str).group(1)) if re.search(r'([+-]?\d+)', rate_str) else 0
        wpm = 150 * (1 + rate_pct / 100)
        duration_ms = int((words / max(wpm, 50)) * 60 * 1000)

        start = _ms_to_srt_time(time_ms)
        end = _ms_to_srt_time(time_ms + duration_ms)

        # Add role prefix for context
        prefix = ""
        if seg.role == Role.HEADER:
            prefix = "[Header] "
        elif seg.role == Role.QUOTE:
            prefix = "[Quote] "
        elif seg.role == Role.AUTHOR:
            prefix = "[Attribution] "

        display = seg.text[:100] + ("..." if len(seg.text) > 100 else "")
        srt_lines.append(f"{idx}\n{start} --> {end}\n{prefix}{display}\n")
        idx += 1
        time_ms += duration_ms + 300  # 300ms gap between segments

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))


def _ms_to_srt_time(ms: int) -> str:
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    mil = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{mil:03d}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 6: CACHING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _cache_key(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}:{text}".encode()).hexdigest()[:16]

def cache_get(text: str, model: str, cache_dir: str) -> Optional[str]:
    path = os.path.join(cache_dir, f"{_cache_key(text, model)}.txt")
    return open(path, "r", encoding="utf-8").read() if os.path.exists(path) else None

def cache_set(text: str, model: str, result: str, cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, f"{_cache_key(text, model)}.txt"), "w", encoding="utf-8") as f:
        f.write(result)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def run_pipeline(args):
    file_path = args.file_path
    if not os.path.isfile(file_path):
        print(f"❌ File not found: {file_path}")
        return

    # Check file extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ['.pdf', '.txt']:
        print(f"❌ Unsupported file type '{ext}'. Please provide a .pdf or .txt file.")
        return

    profile_name = args.profile
    if profile_name not in VOICE_PROFILES:
        print(f"❌ Unknown profile '{profile_name}'. Use --list-profiles.")
        return
    profile = VOICE_PROFILES[profile_name]

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.dirname(os.path.abspath(file_path))
    output_folder = os.path.join(output_dir, f"{base_name}_audio_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    cache_dir = os.path.join(output_folder, CACHE_DIR) if not args.no_cache else None

    # --- Init Gemini ---
    client = None
    skip_ai = args.skip_ai

    os.environ["GEMINI_API_KEY"] = "AIzaSyB5LPqyrbU3VLYTLeB9vFtUPzJ-j7b5Pe4"
    if not skip_ai:
        if genai is None:
            print("⚠️  google-genai not installed. Using local-only mode.")
            skip_ai = True
        elif not os.environ.get("GEMINI_API_KEY"):
            print("⚠️  GEMINI_API_KEY not set. Using local-only mode.\n")
            skip_ai = True
        else:
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            print(f"✅ Gemini ({args.model}) connected.")

    print(f"🎭 Voice profile: {profile_name}\n")

    # ── STEP 1: Extraction ──
    segments = []
    total_pages = 0

    if ext == '.txt':
        print(f"📄 Reading plain text file: {file_path}")
        print("   (Font analysis is disabled for .txt; detecting roles via text formatting)")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        if raw_text.strip():
            segments = text_to_segments(raw_text, page=1)
        total_pages = 1

    elif ext == '.pdf':
        print(f"📄 Extracting PDF with font analysis: {file_path}")
        runs = extract_font_runs(file_path)

        if pdfplumber:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)

        if runs:
            print(f"   {len(runs)} font runs from {total_pages} pages.")
            runs = _remove_margin_runs(runs, total_pages)
            segments = font_runs_to_segments(runs)
            print(f"   {len(segments)} role-tagged segments.")
        else:
            print("   Font extraction unavailable. Falling back to text-only segmentation.")
            # Fallback: extract text and segment by patterns
            if pdfplumber:
                with pdfplumber.open(file_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        raw = page.extract_text(layout=True) or ""
                        if raw.strip():
                            segments.extend(text_to_segments(raw, i + 1))
            else:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                for i, page in enumerate(reader.pages):
                    raw = page.extract_text() or ""
                    if raw.strip():
                        segments.extend(text_to_segments(raw, i + 1))
                total_pages = len(reader.pages)

    if not segments:
        print("❌ No text found.")
        return

    # ── Show role distribution ──
    role_counts = Counter(s.role for s in segments)
    print(f"\n   Role breakdown:")
    for role, count in role_counts.most_common():
        voice_name = profile.get(role, profile[Role.BODY])["voice"].split("-")[-1].replace("Neural", "")
        print(f"     {role:<12} {count:>4} segments → {voice_name}")
    print()

    # ── STEP 2: Remove repeated headers/footers ──
    segments = remove_repeated_segments(segments, total_pages)

    # ── STEP 3: Local NLP cleanup ──
    print("🔧 Local NLP cleanup...")
    for seg in segments:
        seg.text = clean_segment_text(seg.text)
    # Remove empty segments after cleanup
    segments = [s for s in segments if s.text.strip()]
    print(f"   {len(segments)} segments after cleanup.\n")

    # ── STEP 4: Gemini optimization (single attempt, skip all on failure) ──
    if not skip_ai and client:
        print(f"🧠 Gemini optimization...")
        chunk_size = args.chunk_size
        ai_failed = False

        for start in range(0, len(segments), chunk_size):
            if ai_failed:
                break
            chunk = segments[start:start + chunk_size]
            print(f"  📤 Segments {start + 1}–{start + len(chunk)}...")
            try:
                optimized = ai_optimize_segments(chunk, client, args.model)
                for i, opt_seg in enumerate(optimized):
                    if start + i < len(segments):
                        segments[start + i].text = opt_seg.text
                print(f"  ✅ Done.")
            except Exception as e:
                print(f"  ⚠️  Gemini unavailable ({e}). Continuing without AI.")
                ai_failed = True
        print()

    # ── STEP 5: Generate multi-voice audio ──
    # If it's a text file, force it to single-file output since pages don't exist.
    is_single_file = args.single_file or ext == '.txt'

    if is_single_file:
        mp3_path = os.path.join(output_folder, f"{base_name}_full.mp3")
        print(f"🔊 Generating multi-voice audio → {mp3_path}")
        print(f"   ({len(segments)} segments, stitching {len(set(profile[r]['voice'] for r in profile))} voices)...")
        try:
            await stitch_segments_to_mp3(segments, profile, mp3_path, args.srt)
            size_mb = os.path.getsize(mp3_path) / (1024 * 1024)
            print(f"   ✅ Done! ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"   ❌ Failed → {e}")
    else:
        # Split into one audio file per page (PDF only)
        pages: dict[int, list[Segment]] = {}
        for seg in segments:
            pages.setdefault(seg.page, []).append(seg)

        for page_num in sorted(pages):
            page_segs = pages[page_num]
            # Use first header on the page as label, else "Page_N"
            label = next((s.text for s in page_segs if s.role == Role.HEADER), f"Page_{page_num}")
            safe = re.sub(r'[^\w\s-]', '', label)[:40].strip().replace(' ', '_') or f"page_{page_num}"
            mp3_path = os.path.join(output_folder, f"{page_num:03d}_{safe}.mp3")
            print(f"  🔊 [p{page_num}] {label[:40]} → {os.path.basename(mp3_path)}")
            try:
                await stitch_segments_to_mp3(page_segs, profile, mp3_path, args.srt)
            except Exception as e:
                print(f"     ❌ {e}")

    # ── Save script with role annotations ──
    script_path = os.path.join(output_folder, "script.txt")
    with open(script_path, "w", encoding="utf-8") as f:
        for seg in segments:
            voice = profile.get(seg.role, profile[Role.BODY])["voice"]
            f.write(f"[{seg.role}] ({voice})\n")
            f.write(f"  {seg.text}\n\n")

    print(f"\n{'━' * 55}")
    print(f"✅ Done! → {output_folder}/")
    print(f"   📝 Annotated script: {script_path}")
    if args.srt:
        print(f"   📑 SRT subtitles included")
    print(f"   🎭 Profile: {profile_name}")
    print(f"{'━' * 55}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def list_voices():
    voices = await edge_tts.list_voices()
    print(f"\n{'Voice Name':<35} {'Gender':<10} {'Locale'}")
    print("─" * 70)
    for v in sorted(voices, key=lambda x: x["ShortName"]):
        print(f"{v['ShortName']:<35} {v['Gender']:<10} {v['Locale']}")


def list_profiles():
    print("\n🎭 Available Voice Profiles:\n")
    for name, roles in VOICE_PROFILES.items():
        voices_used = sorted(set(v["voice"] for v in roles.values()))
        print(f"  {name}")
        for role_name, cfg in roles.items():
            v = cfg['voice'].split('-')[-1].replace('Neural', '')
            print(f"    {role_name:<12} → {v:<12} rate={cfg['rate']:>5}  vol={cfg['volume']:>5}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="PDF & Text → Audio v3.1: Multi-voice, font-aware, role-based cinematic audio."
    )
    # Changed from pdf_path to file_path to reflect the new functionality
    parser.add_argument("file_path", nargs="?", default=None, help="Path to PDF or TXT file.")
    parser.add_argument("--profile", default="default", help="Voice profile (default, audiobook, cinematic, indian, minimal).")
    parser.add_argument("--skip-ai", action="store_true", help="Local cleanup only.")
    parser.add_argument("--single-file", action="store_true", help="One MP3 for whole PDF (Default for TXT).")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Segments per Gemini call.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--srt", action="store_true", help="Generate SRT subtitles.")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching.")
    parser.add_argument("--list-voices", action="store_true", help="List edge-tts voices.")
    parser.add_argument("--list-profiles", action="store_true", help="List voice profiles.")

    args = parser.parse_args()

    if args.list_voices:
        asyncio.run(list_voices())
        return
    if args.list_profiles:
        list_profiles()
        return
    if args.file_path is None:
        parser.error("Provide a file path (.pdf or .txt), or use --list-voices / --list-profiles.")

    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()