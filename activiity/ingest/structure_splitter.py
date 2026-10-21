"""Structure-aware splitter for Fiches Protocole — v2 (block-packing).

v2 changes vs v1:
  - No content loss. v1 DROPPED any section shorter than `min_chunk_chars`
    and any intro below that size. v2 keeps every character: small content
    is MERGED into neighbouring chunks instead of discarded.
  - Three thresholds instead of one. v1 only had a max size, which produced
    wildly uneven chunks (tiny "weak" one-line chunks next to huge ones).
    v2 packs with target / min / max:
        * too small (< min)  -> merged with neighbouring content
        * too big   (> max)  -> split at the most meaningful boundary
                                (list items first, then sentences)
        * aim        (target) -> greedy pack toward the target size
  - Structure-aware packing, not "split exactly at headings". Within each
    top-level unit (intro / "N principe"), the section text is decomposed
    into semantic blocks (paragraphs, list items) and those blocks are
    packed into target-sized chunks. Sub-sections (Astuce, Exemple, Piège,
    À retenir) pack together, so a one-line Astuce merges with its
    neighbours instead of becoming a weak standalone chunk.
  - Hierarchy preserved in metadata. Each chunk carries `section_path`
    (e.g. "3e principe : … → Astuce : …") so a retrieved chunk knows where
    it came from in the document, plus the existing section_type / header /
    parent_section / principle_index / page fields.

Markers detected (regex bank below, French + a few English alternates):
  - principle headers ("1er principe", "2e principe", "3ème principe")
  - key takeaways      ("À retenir !", "À retenir")
  - tips / conseils    ("Astuce :", "Astuce.", "Conseil :", "Conseil.")
  - examples           ("Exemple :", "Ex :")
  - traps / warnings   ("Piège :")
  - numbered lists     ("3 qualités indispensables", "5 principes pour...")
  - check-lists        ("Check-list", "À retenir !")
  - page boundaries    ("[page N]")

Documents with NO recognised markers fall back to plain block packing
(paragraphs + list items), still size-controlled by the same thresholds —
so prose KB articles no longer become one giant "prose" chunk or an
arbitrary 512-token window.
"""
from __future__ import annotations
import re
from typing import Sequence, Any

from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.node_parser import NodeParser
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager


# ---------------------------------------------------------------------------
# Regex bank — order matters; first match wins for overlapping markers
# ---------------------------------------------------------------------------

_FLAGS = re.IGNORECASE | re.MULTILINE


# Principle header — "1er principe", "2ème principe"
_PRINCIPLE_RE = re.compile(
    r"^(?P<idx>\d+)\s*(?:er|ère|e|ème)?\s+principe\b[^\n]*$",
    _FLAGS,
)


# A bank of "open a new section here" patterns. The 'kind' label becomes
# the chunk's `section_type` metadata.
_SECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (_PRINCIPLE_RE,                                                  "principle"),
    (re.compile(r"^À\s+retenir\s*[!:]?\s*$",                _FLAGS),  "key_takeaways"),
    (re.compile(r"^Check-?list\b[^\n]*$",                   _FLAGS),  "checklist"),
    (re.compile(r"^Astuce\s*[:.]",                          _FLAGS),  "tip"),
    (re.compile(r"^Conseil(?:\s+\d+)?\s*[:.]",              _FLAGS),  "tip"),
    (re.compile(r"^Exemple\s*[:.]",                         _FLAGS),  "example"),
    (re.compile(r"^Pi[èe]ge\s*[:.]",                        _FLAGS),  "trap"),
    # numbered lists — be conservative on the noun set so we don't
    # over-segment prose like "Il y a 3 raisons à cela : ..."
    (re.compile(
        r"^\d+\s+(?:qualités?|principes?|étapes?|critères?|"
        r"règles?|méthodes?|conseils?|cas|axes|registres?|attentes?|"
        r"leviers?|freins?|piliers?|interrogations?|moyens?|attitudes?|"
        r"phases?|sources?|composantes?|outils?|points?\s+de\s+vigilance|"
        r"bonnes?\s+pratiques?|raisons?|mauvaises?\s+habitudes?|signaux)"
        r"\b[^\n]*$",
        _FLAGS,
    ), "criteria_list"),
]


_PAGE_RE = re.compile(r"^\[page\s+(\d+)\]", _FLAGS)

# A line that opens a new block: bullet, or an enumerated item ("3.", "a)")
_LIST_ITEM_RE = re.compile(
    r"^\s*(?:[•▪◦‣●○\-–—*]|\d{1,3}[.)]|[a-z][.)])\s*"
)


# ---------------------------------------------------------------------------
# Marker discovery
# ---------------------------------------------------------------------------

def _find_markers(text: str) -> list[tuple[int, str, str, dict]]:
    """Return [(start_pos, kind, header_line, extra_meta), ...] sorted by position.

    A marker that overlaps an earlier match is dropped (first regex wins).
    """
    found: list[tuple[int, str, str, dict]] = []
    taken: set[int] = set()
    for regex, kind in _SECTION_PATTERNS:
        for m in regex.finditer(text):
            if m.start() in taken:
                continue
            taken.add(m.start())
            line_end = text.find("\n", m.end())
            if line_end == -1:
                line_end = len(text)
            header = text[m.start():line_end].strip()
            extra: dict = {}
            if kind == "principle":
                try:
                    extra["principle_index"] = int(m.group("idx"))
                except (KeyError, ValueError):
                    pass
            found.append((m.start(), kind, header, extra))
    return sorted(found, key=lambda x: x[0])


def _page_at(text: str, pos: int) -> int | None:
    """Return the page number active at character position `pos` (or None)."""
    last = None
    for m in _PAGE_RE.finditer(text):
        if m.start() > pos:
            break
        try:
            last = int(m.group(1))
        except ValueError:
            continue
    return last


# ---------------------------------------------------------------------------
# Block decomposition — the "most meaningful boundaries" for splitting
# ---------------------------------------------------------------------------

def _split_blocks(text: str) -> list[str]:
    """Split section text into semantic blocks: paragraphs and list items.

    - a blank line ends the current paragraph block
    - a line that opens a list item (bullet / "3." / "a)") starts a new block,
      so lists are never split mid-item and consecutive items stay separate
    - wrapped list-item continuation lines stay inside their item block
    - no line is ever dropped
    """
    blocks: list[str] = []
    buf: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            if buf:
                blocks.append("\n".join(buf))
                buf = []
            continue
        if _LIST_ITEM_RE.match(stripped) and buf:
            blocks.append("\n".join(buf))
            buf = [stripped]
        else:
            buf.append(stripped)
    if buf:
        blocks.append("\n".join(buf))
    return [b for b in blocks if b.strip()]


_SENT_BOUNDARY = re.compile(
    r"(?<=[.!?…])\s+(?=[«\"'(\[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ0-9])"
)


def _split_sentences(text: str) -> list[str]:
    """Sentence splitter for oversized blocks (French-aware)."""
    return [p.strip() for p in _SENT_BOUNDARY.split(text) if p.strip()]


def _split_oversized(text: str, max_chars: int) -> list[str]:
    """Split one oversized block by its most meaningful boundaries.

    Sentences first; any single sentence still over budget is hard-wrapped
    at max_chars so no chunk exceeds the cap. Never drops text.
    """
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    cur = ""
    for sent in _split_sentences(text):
        if cur and len(cur) + 1 + len(sent) > max_chars:
            pieces.append(cur.strip())
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        pieces.append(cur.strip())
    final: list[str] = []
    for p in pieces:
        if len(p) <= max_chars:
            final.append(p)
        else:
            final.extend(p[i:i + max_chars] for i in range(0, len(p), max_chars))
    return [p for p in final if p.strip()]


# ---------------------------------------------------------------------------
# Block packing — min / target / max policy
# ---------------------------------------------------------------------------

def _pack_group(
    blocks: list[tuple[str, dict]],
    target_chars: int,
    min_chars: int,
    max_chars: int,
) -> list[tuple[str, list[dict]]]:
    """Pack (block_text, meta) pairs into (chunk_text, [chunk_metas]) chunks.

    Rules:
      * every block first: if > max_chars it is pre-split into ≤ max pieces
      * greedy pack toward target_chars
      * a chunk is only flushed once it has reached at least min_chars
      * when the hard cap is reached with a sub-min buffer, the buffer is
        combined with the incoming unit instead of being flushed alone —
        so no weak standalone chunks are ever emitted; a combined chunk may
        reach up to max_chars + min_chars, far below the embedding cap
      * a trailing buffer below min_chars is merged into the previous chunk
        (never dropped, never a weak standalone chunk)
    """
    units: list[tuple[str, dict]] = []
    for text, meta in blocks:
        if len(text) > max_chars:
            for part in _split_oversized(text, max_chars):
                units.append((part, meta))
        else:
            units.append((text, meta))

    chunks: list[tuple[str, list[dict]]] = []
    buf: list[tuple[str, dict]] = []
    buf_len = 0
    for u in units:
        ulen = len(u[0])
        if buf_len and buf_len + ulen > max_chars:
            if buf_len >= min_chars:
                chunks.append(
                    ("\n\n".join(t for t, _ in buf), [m for _, m in buf])
                )
                buf = [u]
                buf_len = ulen
            else:
                # a sub-min buffer must not become a weak chunk: combine it
                # with the incoming unit instead. Combined size stays under
                # max_chars + min_chars (still far below the embedding cap).
                buf.append(u)
                buf_len += ulen
        elif buf_len and buf_len + ulen > target_chars and buf_len >= min_chars:
            chunks.append(
                ("\n\n".join(t for t, _ in buf), [m for _, m in buf])
            )
            buf = [u]
            buf_len = ulen
        else:
            buf.append(u)
            buf_len += ulen
    if buf:
        joined = "\n\n".join(t for t, _ in buf)
        if chunks and buf_len < min_chars:
            prev_text, prev_metas = chunks[-1]
            # weak chunks hurt retrieval more than a slightly-long chunk;
            # a merged remainder may reach up to max_chars + min_chars
            chunks[-1] = (
                prev_text + "\n\n" + joined,
                prev_metas + [m for _, m in buf],
            )
        else:
            chunks.append((joined, [m for _, m in buf]))
    return chunks


def _group_chars(group: list[tuple[str, dict]]) -> int:
    return sum(len(text) for text, _ in group)


def _chunk_meta(metas: list[dict], base_meta: dict) -> dict:
    """Merge per-block section metadata into one chunk metadata dict.

    The chunk keeps the primary section fields and adds a `section_path`
    describing the full hierarchy it covers. If the chunk starts with the
    document title (intro) but also contains real section content, the
    real section label wins as primary.
    """
    primary = metas[0]
    if primary.get("section_type") in ("intro",):
        for m in metas:
            if m.get("section_type") not in ("intro",):
                primary = m
                break

    meta = dict(base_meta)
    for key in ("section_type", "header", "principle_index",
                "parent_section", "page"):
        value = primary.get(key)
        if value is not None:
            meta[key] = value

    path: list[str] = []
    for m in metas:
        header = m.get("header")
        if header and header not in path:
            path.append(header[:200])
    if path:
        meta["section_path"] = " → ".join(path[:5])

    types: list[str] = []
    for m in metas:
        st = m.get("section_type")
        if st and st not in types:
            types.append(st)
    meta["section_types"] = types

    sections = {(m.get("section_type"), m.get("header")) for m in metas}
    meta["n_sections"] = len(sections)
    meta["n_blocks"] = len(metas)
    return meta


# ---------------------------------------------------------------------------
# Core: split a text into structured chunks with metadata
# ---------------------------------------------------------------------------

def split_by_structure(
    text: str,
    base_meta: dict,
    *,
    target_chunk_chars: int = 1100,
    min_chunk_chars: int = 300,
    max_chunk_chars: int = 2000,
) -> list[tuple[str, dict]]:
    """Split `text` into (chunk_text, chunk_metadata) tuples.

    base_meta is inherited (e.g. ua_id, source_path, doc_type).

    "N principe" units are hard boundaries — chunks never cross them. Units
    too small to stand alone are merged with their neighbours (including the
    document title). Everything inside a unit packs together toward the
    target size, so small sub-sections merge with their neighbours and large
    sections are split at list-item / sentence boundaries.
    """
    markers = _find_markers(text)

    # ---- Build sections: [(section_text, section_meta)] -----------------
    if not markers:
        sections: list[tuple[str, dict]] = [
            (text, {**base_meta, "section_type": "prose"})
        ]
    else:
        sections = []

        if markers[0][0] > 0:
            intro = text[:markers[0][0]].strip()
            if intro:
                meta = {**base_meta, "section_type": "intro"}
                page = _page_at(text, 0)
                if page is not None:
                    meta["page"] = page
                sections.append((intro, meta))

        current_principle: str | None = None
        current_principle_idx: int | None = None
        for i, (start, kind, header, extra) in enumerate(markers):
            end = markers[i + 1][0] if i + 1 < len(markers) else len(text)
            section_text = text[start:end].strip()
            if not section_text:
                continue

            if kind == "principle":
                current_principle = header
                current_principle_idx = extra.get("principle_index")

            meta: dict = {
                **base_meta,
                "section_type": kind,
                "header": header[:200],
                **extra,
            }
            if kind != "principle" and current_principle:
                meta["parent_section"] = current_principle[:200]
                if current_principle_idx is not None and "principle_index" not in meta:
                    meta["principle_index"] = current_principle_idx

            page = _page_at(text, start)
            if page is not None:
                meta["page"] = page

            sections.append((section_text, meta))

    # ---- Group blocks into top-level units ------------------------------
    # "N principe" headings are hard boundaries — chunks never cross them.
    # Everything before the first principle (title/intro) joins that first
    # unit; sub-sections (Astuce, Exemple, ...) pack inside their unit.
    groups: list[list[tuple[str, dict]]] = []
    for section_text, meta in sections:
        if meta["section_type"] == "principle" or not groups:
            groups.append([])
        for block in _split_blocks(section_text):
            groups[-1].append((block, meta))

    # ---- Merge units too small to stand alone (never drop, no weak chunks)
    merged_groups: list[list[tuple[str, dict]]] = []
    for group in groups:
        if merged_groups and _group_chars(group) < min_chunk_chars:
            merged_groups[-1].extend(group)
        else:
            merged_groups.append(group)
    if len(merged_groups) > 1 and _group_chars(merged_groups[0]) < min_chunk_chars:
        merged_groups[1] = merged_groups[0] + merged_groups[1]
        merged_groups.pop(0)
    groups = merged_groups

    # ---- Pack each group toward the target size -------------------------
    out: list[tuple[str, dict]] = []
    for group in groups:
        if not group:
            continue
        for chunk_text, chunk_metas in _pack_group(
            group,
            target_chars=target_chunk_chars,
            min_chars=min_chunk_chars,
            max_chars=max_chunk_chars,
        ):
            out.append((chunk_text, _chunk_meta(chunk_metas, base_meta)))
    return out


# ---------------------------------------------------------------------------
# LlamaIndex TransformComponent — drop-in replacement for SentenceSplitter
# ---------------------------------------------------------------------------

class StructureAwareSplitter(NodeParser):
    """Drop-in transformation for IngestionPipeline.

    Use exactly like SentenceSplitter:
        pipeline = IngestionPipeline(transformations=[
            StructureAwareSplitter(),
            embed_model,
        ])

    Thresholds (characters, ~2.2 chars/token for French):
        target_chunk_chars — the size we pack toward (default 1100 ≈ 500 tok)
        min_chunk_chars    — below this, merge with neighbours (default 300)
        max_chunk_chars    — above this, split at list/sentence boundaries
                              (default 2000 ≈ 900 tok, under the 8192 cap)
    """

    target_chunk_chars: int = Field(default=1100)
    min_chunk_chars: int = Field(default=300)
    max_chunk_chars: int = Field(default=2000)
    # Kept for backward compatibility with callers that passed the old
    # token-based fallback knobs (e.g. baseline/index.py). No longer used.
    fallback_chunk_size: int = Field(default=512)
    fallback_overlap: int = Field(default=64)
    _callback_manager: CallbackManager = PrivateAttr()

    def __init__(
        self,
        target_chunk_chars: int = 1100,
        min_chunk_chars: int = 300,
        max_chunk_chars: int = 2000,
        fallback_chunk_size: int = 512,
        fallback_overlap: int = 64,
        callback_manager: CallbackManager | None = None,
    ) -> None:
        super().__init__(
            target_chunk_chars=target_chunk_chars,
            min_chunk_chars=min_chunk_chars,
            max_chunk_chars=max_chunk_chars,
            fallback_chunk_size=fallback_chunk_size,
            fallback_overlap=fallback_overlap,
            callback_manager=callback_manager or CallbackManager([]),
        )

    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> list[BaseNode]:
        out: list[BaseNode] = []
        for node in nodes:
            text = node.get_content()
            base_meta = dict(node.metadata) if node.metadata else {}
            for chunk_text, chunk_meta in split_by_structure(
                text, base_meta,
                target_chunk_chars=self.target_chunk_chars,
                min_chunk_chars=self.min_chunk_chars,
                max_chunk_chars=self.max_chunk_chars,
            ):
                out.append(TextNode(text=chunk_text, metadata=chunk_meta))
        return out

    def _get_nodes_from_node(self, node: BaseNode) -> list[BaseNode]:
        """Legacy method for older LlamaIndex versions."""
        return self._parse_nodes([node])
