"""Lectura de los PDFs del corpus: texto, título, autores y abstract.

El abstract se localiza buscando el encabezado "Abstract" (o "Resumen") y
cortando en la primera sección siguiente (Keywords, Introduction, etc.).
Título y autores se estiman con heurísticas; lo que no se pueda extraer bien se
corrige a mano en data/overrides.json.
"""
import re

from pypdf import PdfReader

_LIGATURES = {"\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi", "\ufb04": "ffl"}

_ABSTRACT_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:abstract|resumen)[ \t]*[:.\-\u2014\u2013]*[ \t]*(?P<body>.*?)"
    r"(?=\n[ \t]*(?:keywords?|key[ \t]*words|index[ \t]+terms|palabras[ \t]+clave|"
    r"ccs[ \t]+concepts|(?:1|i)[ \t]*[.)]?[ \t]+introduction|introduction)\b|\Z)",
    re.IGNORECASE | re.DOTALL,
)

_NOISE_RE = re.compile(
    r"@|universit|institut|department|school of|laborator|research|google|microsoft|"
    r"openai|meta ai|deepmind|inc\.|corp|college|\d{4}|arxiv|http|www\.",
    re.IGNORECASE,
)


def clean_text(s: str) -> str:
    """Normaliza espacios, ligaduras y palabras cortadas por fin de línea."""
    for lig, repl in _LIGATURES.items():
        s = s.replace(lig, repl)
    s = s.replace("\u00ad", "")
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)   # une palabras cortadas con guion al final de línea
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def read_first_pages(path: str, max_pages: int = 2) -> tuple[str, str | None]:
    """Devuelve (texto de las primeras páginas, título de los metadatos o None)."""
    reader = PdfReader(path)
    meta_title = reader.metadata.title if reader.metadata else None
    pages = [(p.extract_text() or "") for p in reader.pages[:max_pages]]
    return "\n".join(pages), meta_title


def find_abstract(text: str) -> tuple[str | None, int]:
    """Devuelve (abstract limpio o None, posición donde empieza el encabezado)."""
    m = _ABSTRACT_RE.search(text)
    if not m:
        return None, -1
    body = clean_text(m.group("body"))
    return (body or None), m.start()


def _plausible_title(t: str | None) -> bool:
    if not t:
        return False
    t = t.strip()
    low = t.lower()
    if len(t) < 10 or low.endswith((".pdf", ".dvi", ".tex", ".doc", ".docx")):
        return False
    return not low.startswith(("microsoft word", "untitled", "paper", "arxiv"))


def guess_title(text: str, meta_title: str | None) -> tuple[str, str]:
    """Devuelve (título, fuente) donde fuente es 'metadatos' o 'primera linea'."""
    if _plausible_title(meta_title):
        return clean_text(meta_title), "metadatos"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return (clean_text(lines[0]) if lines else ""), "primera linea"


def guess_authors(front_text: str, title: str) -> str:
    """Estima los autores a partir del texto previo al abstract (muy aproximado)."""
    candidates = []
    for line in front_text.splitlines():
        ln = line.strip()
        if not ln or len(ln) > 120 or _NOISE_RE.search(ln):
            continue
        if ln in title or title.startswith(ln):
            continue
        candidates.append(ln)
    return "; ".join(candidates[:6])