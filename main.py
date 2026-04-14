"""
PDF-to-Audio Pipeline v3 — Multi-Voice Cinematic Edition
==========================================================
Pipeline: pdfplumber (extract + font analysis) → Role detection → Gemini (polish)
          → edge-tts (multi-voice) → MP3 stitching

NEW IN v3:
  • Font-aware extraction — reads each character's font name & size from the PDF
    to detect bold, italic, headings, and body text at the source level.
  • Role-based voice casting — headers, body, emphasis (bold), asides (italic),
    quotes, author attributions, and list items each get a distinct voice/tone.
  • Automatic pattern detection — "-- Author Name", blockquotes, parenthetical
    asides, numbered lists, and em-dash interjections are all tagged.
  • Segment-level audio generation — each segment is spoken in its assigned
    voice, then stitched into a seamless MP3 with natural pauses between roles.
  • Configurable voice profiles — swap voices via a simple JSON config or CLI.

Requirements:
    pip install pdfplumber edge-tts google-genai

Usage:
    export GEMINI_API_KEY="AIza..."
    python pdf_to_audio_v3.py "file.pdf" --single-file
    python pdf_to_audio_v3.py "file.pdf" --skip-ai --profile cinematic
    python pdf_to_audio_v3.py "file.pdf" --profile audiobook --srt
    python pdf_to_audio_v3.py --list-voices
    python pdf_to_audio_v3.py --list-profiles
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

TTS_OPTIMIZER_PROMPT = """You are an Audio Script Editor optimizing PDF-extracted text for Text-to-Speech.

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
# STEP 1: FONT-AWARE EXTRACTION
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


def _is_bold(fontname: str) -> bool:
    fn = fontname.lower()
    return any(k in fn for k in ("bold", "heavy", "black", "demi", "semibold"))


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
        norm = re.sub(r'\d+', '#', r.text.strip())
        if not norm:
            continue
        margin_texts.setdefault(norm, set()).add(r.page)

    # Which normalized strings appear on enough pages?
    repeated = set()
    for norm, pages in margin_texts.items():
        if len(pages) / total_pages >= 0.30:
            repeated.add(norm)

    if not repeated:
        return runs

    filtered = []
    removed = 0
    for r in runs:
        norm = re.sub(r'\d+', '#', r.text.strip())
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
# STEP 2b: FALLBACK — TEXT-ONLY SEGMENTATION (no font info)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def text_to_segments(text: str, page: int = 0) -> list[Segment]:
    """
    When pdfplumber is unavailable or font extraction fails,
    segment text purely by regex patterns.
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

    text_page_counts = Counter()
    for seg in segments:
        normalized = re.sub(r'\d+', '#', seg.text.strip())
        text_page_counts[(normalized, seg.page)] = 1

    # Count across pages (not occurrences)
    text_counts = Counter()
    for (norm_text, _), _ in text_page_counts.items():
        text_counts[norm_text] += 1

    repeated = {t for t, c in text_counts.items() if c / total_pages >= 0.4}

    if not repeated:
        return segments

    filtered = []
    removed = 0
    for seg in segments:
        normalized = re.sub(r'\d+', '#', seg.text.strip())
        if normalized in repeated:
            removed += 1
            continue
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


def clean_segment_text(text: str) -> str:
    """Apply all local NLP cleanup to a segment's text."""
    # Fix hyphenation
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

    # Remove PDF encoding artifacts (e.g. bullet glyphs rendered as (cid:127))
    text = re.sub(r'\(cid:\d+\)', '', text)

    # Remove citations
    text = re.sub(r'\s*\[\d+(?:[,\-–]\s*\d+)*\]', '', text)
    text = re.sub(r'\s*\([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][a-z]+)*,?\s*\d{4}[a-z]?\)', '', text)
    text = re.sub(r'[†‡§¶]', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # Remove standalone page numbers
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)

    # Expand abbreviations
    for pat, rep in ABBREVIATIONS.items():
        text = re.sub(pat, rep, text)

    # Expand units
    for pat, rep in UNITS.items():
        text = re.sub(pat, rep, text)

    # Expand symbols
    text = re.sub(r'\s*&\s*', ' and ', text)
    text = re.sub(r'@', ' at ', text)
    text = re.sub(r'(?<!\w)#(\d+)', r'number \1', text)
    text = re.sub(r'~(\d)', r'approximately \1', text)
    text = re.sub(r'≥', 'greater than or equal to ', text)
    text = re.sub(r'≤', 'less than or equal to ', text)
    text = re.sub(r'→', 'leads to ', text)

    # Expand currencies
    def _money(m):
        sym = m.group(1)
        amt = m.group(2)
        suf = (m.group(3) or "").upper()
        cur = {"$": "dollars", "£": "pounds", "€": "euros", "₹": "rupees"}.get(sym, "dollars")
        mul = {"K": "thousand", "M": "million", "B": "billion", "T": "trillion"}.get(suf, "")
        return f"{amt} {mul} {cur}".strip() if mul else f"{amt} {cur}"
    text = re.sub(r'([$£€₹])\s*(\d[\d,]*\.?\d*)\s*([KkMmBbTt])?(?=\s|$|[,.\)])', _money, text)

    # Expand ordinals
    ordinals = {'1st': 'first', '2nd': 'second', '3rd': 'third', '4th': 'fourth',
                '5th': 'fifth', '10th': 'tenth', '20th': 'twentieth', '21st': 'twenty-first'}
    for num, word in ordinals.items():
        text = re.sub(rf'\b{num}\b', word, text)

    # Expand dates
    months = {'01':'January','02':'February','03':'March','04':'April','05':'May',
              '06':'June','07':'July','08':'August','09':'September','10':'October',
              '11':'November','12':'December'}
    def _iso(m):
        return f"{months.get(m.group(2), m.group(2))} {int(m.group(3))}, {m.group(1)}"
    text = re.sub(r'\b(\d{4})[-/](\d{2})[-/](\d{2})\b', _iso, text)

    # Em-dash → comma
    text = re.sub(r'\s*[—–]\s*', ', ', text)

    # Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', ' ', text)

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
        "Here is role-tagged PDF text. Each section starts with a role marker like [HEADER], [BODY], etc.\n"
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

    try:
        for idx, seg in enumerate(segments):
            if seg.role == Role.PAUSE:
                continue

            # Skip segments with no speakable text
            speakable = re.sub(r'[^a-zA-Z0-9]', '', seg.text)
            if not speakable:
                continue

            voice_cfg = profile.get(seg.role, profile[Role.BODY])
            chunk_path = os.path.join(tmp_dir, f"{idx:05d}_segment.mp3")

            # Generate this segment's audio — skip on failure, never crash the page
            try:
                await generate_segment_audio(
                    seg.text, voice_cfg["voice"], voice_cfg["rate"], voice_cfg["volume"], chunk_path
                )
            except Exception:
                continue

            # Each edge-tts utterance already has ~300ms natural leading/trailing silence,
            # so no separate pause file is needed — it would double the gap.
            chunk_paths.append(chunk_path)

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
    pdf_path = args.pdf_path
    if not os.path.isfile(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return

    profile_name = args.profile
    if profile_name not in VOICE_PROFILES:
        print(f"❌ Unknown profile '{profile_name}'. Use --list-profiles.")
        return
    profile = VOICE_PROFILES[profile_name]

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_dir = os.path.dirname(os.path.abspath(pdf_path))
    output_folder = os.path.join(pdf_dir, f"{base_name}_audio_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    cache_dir = os.path.join(output_folder, CACHE_DIR) if not args.no_cache else None

    # --- Init Gemini ---
    client = None
    skip_ai = args.skip_ai
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

    # ── STEP 1: Font-aware extraction ──
    print(f"📄 Extracting with font analysis: {pdf_path}")
    runs = extract_font_runs(pdf_path)

    total_pages = 0
    if pdfplumber:
        with pdfplumber.open(pdf_path) as pdf:
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
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    raw = page.extract_text(layout=True) or ""
                    if raw.strip():
                        segments = text_to_segments(raw, i + 1)
        else:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            segments = []
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
    if args.single_file:
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
        # Split into one audio file per page
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
        description="PDF → Audio v3: Multi-voice, font-aware, role-based cinematic audio."
    )
    parser.add_argument("pdf_path", nargs="?", default=None, help="Path to PDF.")
    parser.add_argument("--profile", default="default", help="Voice profile (default, audiobook, cinematic, indian, minimal).")
    parser.add_argument("--skip-ai", action="store_true", help="Local cleanup only.")
    parser.add_argument("--single-file", action="store_true", help="One MP3 for whole PDF.")
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
    if args.pdf_path is None:
        parser.error("Provide a PDF path, or use --list-voices / --list-profiles.")

    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
