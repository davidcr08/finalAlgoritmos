"""Lectura de los PDFs del corpus: texto, título, autores y abstract.

El abstract se localiza buscando el encabezado "Abstract" (también con letras
separadas, "A B S T R A C T", como en varias revistas de Elsevier) y cortando en
la primera sección siguiente (Keywords, Introduction, etc.). Cuando un artículo
no encaja, se ajusta con reglas por artículo en data/overrides.json.
"""
import re
import unicodedata

import pymupdf

_START = r"(?<![a-z])a\s*b\s*s\s*t\s*r\s*a\s*c\s*t(?![a-z])"

_END_MARKERS = (
    r"(?:k\s*e\s*y\s*w\s*o\s*r\s*d\s*s|key\s+words|index\s+terms|palabras\s+clave"
    r"|article\s+history|ccs\s+concepts|(?:1|i)\.?\s+introduction|introduction)"
)

_NOISE_RE = re.compile(
    r"@|universit|institut|department|school of|laborator|research|google|microsoft|"
    r"openai|meta ai|deepmind|inc\.|corp|college|\d{4}|arxiv|http|www\.|"
    r"open access|creativecommons|licen[sc]e|review article|article info|keywords?\b|"
    r"citation:|doi|nc-nd|nc/4|contents lists|sciencedirect|journal|received|accepted|"
    r"copyright|available online|elsevier|springer|wiley|mdpi|academic editor|"
    r"all rights|published|\u00a9|"
    r"a\s*r\s*t\s*i\s*c\s*l\s*e\s*i\s*n\s*f\s*o|a\s*b\s*s\s*t\s*r\s*a\s*c\s*t",
    re.IGNORECASE,
)

# Línea de afiliación con marcador de una letra: "a Centre for...", "b Escuela..."
_AFFILIATION_RE = re.compile(r"^[a-z]\s+[A-Z]")


def normalize_text(text: str) -> str:
    """NFKC (ligaduras tipográficas -> letras normales), guiones suaves y palabras cortadas por guion."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)   # caracteres de ancho cero
    text = re.sub(r"\u00ad\s*", "", text)
    text = re.sub(r"([a-z])-[ \t]*\n[ \t]*([a-z])", r"\1\2", text)
    return text


def clean_text(s: str) -> str:
    """Normaliza y colapsa los espacios en uno solo."""
    return re.sub(r"\s+", " ", normalize_text(s)).strip()


def read_first_pages(path: str, max_pages: int = 2) -> tuple[str, str | None]:
    """Devuelve (texto de las primeras páginas, título de los metadatos o None)."""
    with pymupdf.open(path) as doc:
        meta_title = (doc.metadata or {}).get("title")
        text = "\n".join(doc[i].get_text() for i in range(min(max_pages, len(doc))))
    return text, meta_title


def find_abstract(
    text: str, end_phrase: str | None = None, start_trim: str | None = None
) -> tuple[str | None, int]:
    """Devuelve (abstract limpio o None, posición del encabezado en el texto normalizado).

    end_phrase: frase con la que empieza el texto que sigue al abstract, para
                artículos sin marca de cierre (Keywords/Introduction).
    start_trim: último fragmento (inclusive) de lo que precede al abstract, para
                artículos con las palabras clave antes del abstract.
    """
    text = normalize_text(text)
    end = rf"\n\s*{_END_MARKERS}\b"
    if end_phrase:
        phrase = r"\s+".join(re.escape(w) for w in end_phrase.split())
        end = rf"(?:{end}|{phrase})"
    pattern = rf"{_START}\s*[:\-\u2014\u2013.]?\s*(?P<body>.+?)(?={end}|\Z)"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None, -1
    body = re.sub(r"\s+", " ", m.group("body")).strip()
    if start_trim:
        idx = body.lower().find(start_trim.lower())
        if idx != -1:
            body = body[idx + len(start_trim):].lstrip(" .,:;").strip()
    return (body or None), m.start()


def _plausible_title(t: str | None) -> bool:
    if not t:
        return False
    t = t.strip()
    low = t.lower()
    if len(t) < 10 or len(t.split()) < 3:
        return False
    if low.endswith((".pdf", ".dvi", ".tex", ".doc", ".docx")):
        return False
    if re.search(r"_proof|\d+\.\.\d+|\.indd|\.qxd", low):
        return False
    return not low.startswith(("microsoft word", "untitled", "paper", "arxiv"))


def guess_title(text: str, meta_title: str | None) -> tuple[str, str]:
    """Devuelve (título, fuente) donde fuente es 'metadatos' o 'primera linea'."""
    if _plausible_title(meta_title):
        return clean_text(meta_title), "metadatos"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return (clean_text(lines[0]) if lines else ""), "primera linea"


def _key(s: str) -> str:
    return re.sub(r"[\W_]+", "", s.lower())


def guess_authors(front_text: str, title: str) -> str:
    """Estima los autores a partir del texto previo al abstract (muy aproximado)."""
    tkey = _key(title)
    candidates = []
    for line in front_text.splitlines():
        ln = line.strip()
        if not ln or len(ln) > 120 or _NOISE_RE.search(ln) or _AFFILIATION_RE.match(ln):
            continue
        k = _key(ln)
        if len(k) >= 8 and tkey and (k in tkey or tkey in k):
            continue
        candidates.append(ln)
    return "; ".join(candidates[:6])