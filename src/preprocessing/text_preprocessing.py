"""Preprocesamiento de texto compartido por todos los requerimientos.

Pipeline: minúsculas -> tokenización (solo letras) -> filtro de longitud ->
eliminación de stopwords. No se aplica stemming ni lematización, para que los
tokens se puedan buscar tal cual en los vectores de palabras preentrenados.
"""
import json
import re
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1]+")

STOPWORDS_EN = frozenset("""
a about above after again against all am an and any are aren as at be because been
before being below between both but by can cannot could did do does doing down during
each few for from further had has have having he her here hers herself him himself his
how i if in into is it its itself just let me more most my myself no nor not of off on
once only or other our ours ourselves out over own same she should so some such than
that the their theirs them themselves then there these they this those through to too
under until up very was we were what when where which while who whom why will with
would you your yours yourself yourselves also however thus therefore may might must
shall via et al etc eg ie using used use within without among across toward towards
per upon whether either neither whereas hence
""".split())

STOPWORDS_ES = frozenset("""
a al algo algunas algunos ante antes como con contra cual cuando de del desde donde
durante e el ella ellas ellos en entre era erais eran eras eres es esa esas ese eso esos
esta estaba estado estar este esto estos fue fueron ha han hasta hay la las le les lo
los mas me mi mis mucho muy nada ni no nos nosotros o os otra otras otro otros para pero
poco por porque que quien quienes se sea sean ser si sido sin sobre solo su sus tambien
tan te tiene tienen todo todos tu tus un una uno unos y ya yo
""".split())

_STOPWORDS_BY_LANG = {"en": STOPWORDS_EN, "es": STOPWORDS_ES}


def tokenize(text: str) -> list[str]:
    """Minúsculas y tokens formados solo por letras (descarta números y signos)."""
    return _TOKEN_RE.findall(text.lower())


def preprocess(
    text: str,
    remove_stopwords: bool = True,
    languages: tuple[str, ...] = ("en",),
    min_len: int = 2,
) -> list[str]:
    """Devuelve la lista de tokens limpios de un texto."""
    stop = frozenset().union(*(_STOPWORDS_BY_LANG[lang] for lang in languages))
    tokens = [t for t in tokenize(text) if len(t) >= min_len]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop]
    return tokens


def load_corpus(path: str = "data/processed/corpus.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def preprocess_corpus(corpus: list[dict], **kwargs) -> dict[str, list[str]]:
    """id del artículo -> tokens de su abstract."""
    return {doc["id"]: preprocess(doc["abstract"], **kwargs) for doc in corpus}