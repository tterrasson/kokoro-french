"""German G2P: text normalization + espeak-ng phonemization.

normalize_text_de() expands numbers, dates, times, currency, and
abbreviations so espeak-ng receives clean spelled-out text.

DEG2P wraps normalize_text_de() + EspeakG2P for use in KPipeline.
"""

from typing import Tuple
import re

# ── cardinal numbers ─────────────────────────────────────────────────────────

_ONES = [
    "",
    "ein",
    "zwei",
    "drei",
    "vier",
    "fünf",
    "sechs",
    "sieben",
    "acht",
    "neun",
    "zehn",
    "elf",
    "zwölf",
    "dreizehn",
    "vierzehn",
    "fünfzehn",
    "sechzehn",
    "siebzehn",
    "achtzehn",
    "neunzehn",
]
_TENS = [
    "",
    "",
    "zwanzig",
    "dreißig",
    "vierzig",
    "fünfzig",
    "sechzig",
    "siebzig",
    "achtzig",
    "neunzig",
]


def _int_to_de(n, standalone=True):
    """Convert integer to German words.

    standalone=True returns "eins" for 1, standalone=False returns "ein"
    (used in composition: einhundert, eintausend).
    """
    if n < 0:
        return "minus " + _int_to_de(-n)
    if n == 0:
        return "null"
    if n == 1:
        return "eins" if standalone else "ein"
    if n < 20:
        return _ONES[n]
    if n < 100:
        ones, tens = n % 10, n // 10
        if ones:
            return _ONES[ones] + "und" + _TENS[tens]
        return _TENS[tens]
    if n < 1_000:
        h, r = n // 100, n % 100
        return _ONES[h] + "hundert" + (_int_to_de(r, standalone=False) if r else "")
    if n < 1_000_000:
        t, r = n // 1_000, n % 1_000
        prefix = _int_to_de(t, standalone=False) if t != 1 else "ein"
        return prefix + "tausend" + (_int_to_de(r, standalone=False) if r else "")
    if n < 1_000_000_000:
        m, r = n // 1_000_000, n % 1_000_000
        word = (
            "eine Million" if m == 1 else _int_to_de(m, standalone=False) + " Millionen"
        )
        return word + (" " + _int_to_de(r, standalone=False) if r else "")
    b, r = n // 1_000_000_000, n % 1_000_000_000
    word = (
        "eine Milliarde" if b == 1 else _int_to_de(b, standalone=False) + " Milliarden"
    )
    return word + (" " + _int_to_de(r, standalone=False) if r else "")


# ── ordinals ─────────────────────────────────────────────────────────────────

_ORD_IRREG = {1: "erst", 2: "zweit", 3: "dritt", 7: "siebt", 8: "acht"}


def _ordinal_stem_de(n):
    """Ordinal stem without inflection suffix."""
    if n in _ORD_IRREG:
        return _ORD_IRREG[n]
    return _int_to_de(n, standalone=False) + ("t" if n < 20 else "st")


# ── years ────────────────────────────────────────────────────────────────────


def _year_de(n):
    """German year pronunciation: 1985 -> neunzehnhundertfünfundachtzig."""
    if 1100 <= n <= 1999:
        c, r = n // 100, n % 100
        return (
            _int_to_de(c, standalone=False)
            + "hundert"
            + (_int_to_de(r, standalone=False) if r else "")
        )
    return _int_to_de(n)


# ── month names ──────────────────────────────────────────────────────────────

_MONTHS = [
    "",
    "Januar",
    "Februar",
    "März",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Dezember",
]

# ── currency ─────────────────────────────────────────────────────────────────

_CURRENCY = {"€": "Euro", "$": "Dollar", "£": "Pfund", "¥": "Yen"}


def _currency_repl(sym, num):
    word = _CURRENCY.get(sym, sym)
    cleaned = num.replace(".", "").replace(",", ".")
    try:
        val = float(cleaned)
    except ValueError:
        return sym + num
    euros = int(val)
    cents = round((val - euros) * 100)
    if cents == 0:
        return _int_to_de(euros) + " " + word
    return _int_to_de(euros) + " " + word + " und " + _int_to_de(cents) + " Cent"


# ── text normalization ───────────────────────────────────────────────────────


def normalize_text_de(text):
    """Normalize German text for TTS: expand numbers, dates, times, currency, abbreviations."""
    if not text:
        return text

    # 1. Quotes -> ASCII
    text = text.replace("\u201e", '"').replace("\u201c", '"')  # „ "
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # ' '
    text = text.replace("\u00ab", '"').replace("\u00bb", '"')  # « »
    text = text.replace("\u2039", '"').replace("\u203a", '"')  # ‹ ›

    # 2. Non-breaking whitespace
    text = re.sub(r"[^\S \n]", " ", text)

    # 3. Abbreviations
    text = re.sub(r"\bDr\.(?=\s)", "Doktor", text)
    text = re.sub(r"\bProf\.(?=\s)", "Professor", text)
    text = re.sub(r"\bHr\.(?=\s)", "Herr ", text)
    text = re.sub(r"\bFr\.(?=\s[A-ZÄÖÜ])", "Frau", text)
    text = re.sub(r"\bDipl\.\s*-?\s*Ing\.", "Diplom-Ingenieur", text)
    text = re.sub(r"\bStr\.(?=\s)", "Straße", text)
    text = re.sub(r"\bNr\.(?=\s*\d)", "Nummer", text)
    text = re.sub(r"\bTel\.(?=\s)", "Telefon", text)
    text = re.sub(r"\bAbt\.(?=\s)", "Abteilung", text)
    text = re.sub(r"\bGmbH\b", "Gesellschaft mit beschränkter Haftung", text)
    text = re.sub(r"\bAG\b(?=[\s,.]|$)", "Aktiengesellschaft", text)
    text = re.sub(r"\bz\.\s*B\.", "zum Beispiel", text, flags=re.IGNORECASE)
    text = re.sub(r"\bd\.\s*h\.", "das heißt", text, flags=re.IGNORECASE)
    text = re.sub(r"\bu\.\s*a\.", "unter anderem", text, flags=re.IGNORECASE)
    text = re.sub(r"\bbzw\.", "beziehungsweise", text, flags=re.IGNORECASE)
    text = re.sub(r"\busw\.", "und so weiter", text, flags=re.IGNORECASE)
    text = re.sub(r"\betc\.", "et cetera", text, flags=re.IGNORECASE)
    text = re.sub(r"\bca\.", "circa", text, flags=re.IGNORECASE)
    text = re.sub(r"\bvgl\.", "vergleiche", text, flags=re.IGNORECASE)
    text = re.sub(r"\binkl\.", "inklusive", text, flags=re.IGNORECASE)
    text = re.sub(r"\bexkl\.", "exklusive", text, flags=re.IGNORECASE)
    text = re.sub(r"\bggf\.", "gegebenenfalls", text, flags=re.IGNORECASE)
    text = re.sub(r"\bi\.\s*d\.\s*R\.", "in der Regel", text, flags=re.IGNORECASE)
    text = re.sub(r"\bo\.\s*ä\.", "oder ähnliches", text, flags=re.IGNORECASE)
    text = re.sub(r"\bu\.\s*U\.", "unter Umständen", text, flags=re.IGNORECASE)
    # Month abbreviations
    for abbr, full in [
        ("Jan", "Januar"),
        ("Feb", "Februar"),
        ("Mär", "März"),
        ("Apr", "April"),
        ("Jun", "Juni"),
        ("Jul", "Juli"),
        ("Aug", "August"),
        ("Sep", "September"),
        ("Okt", "Oktober"),
        ("Nov", "November"),
        ("Dez", "Dezember"),
    ]:
        text = re.sub(rf"\b{abbr}\.(?=\s)", full, text)

    # 4. Currency (symbol before or after amount)
    csym = r"[€$£¥]"
    text = re.sub(
        rf"({csym})\s*(\d[\d.,]*)",
        lambda m: _currency_repl(m.group(1), m.group(2)),
        text,
    )
    text = re.sub(
        rf"(\d[\d.,]*)\s*({csym})",
        lambda m: _currency_repl(m.group(2), m.group(1)),
        text,
    )

    # 5. Times (HH:MM)
    def _time_repl(m):
        h, mi = int(m.group(1)), int(m.group(2))
        return _int_to_de(h) + " Uhr" + (" " + _int_to_de(mi) if mi else "")

    text = re.sub(r"\b(\d{1,2}):(\d{2})(?:\s*Uhr)?", _time_repl, text)

    # 6. Full dates (DD.MM.YYYY)
    def _date_repl(m):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if d < 1 or d > 31 or mo < 1 or mo > 12:
            return m.group(0)
        return _ordinal_stem_de(d) + "e " + _MONTHS[mo] + " " + _year_de(y)

    text = re.sub(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", _date_repl, text)

    # 7. Ordinals mid-sentence (e.g. "am 3. Mai") -- only 1-2 digit numbers
    text = re.sub(
        r"(?<!\d)(\d{1,2})\.\s",
        lambda m: _ordinal_stem_de(int(m.group(1))) + "e ",
        text,
    )

    # 8. Standalone years (1100-2099)
    def _year_repl(m):
        n = int(m.group(1))
        return _year_de(n) if 1100 <= n <= 2099 else _int_to_de(n)

    text = re.sub(r"\b(\d{4})\b", _year_repl, text)

    # 9. German-format numbers: 1.234.567 or 1.234,56
    def _grouped_num_repl(m):
        cleaned = m.group(0).replace(".", "").replace(",", ".")
        try:
            val = float(cleaned)
        except ValueError:
            return m.group(0)
        if val == int(val):
            return _int_to_de(int(val))
        ip, fp = cleaned.split(".")
        return (
            _int_to_de(int(ip)) + " Komma " + " ".join(_int_to_de(int(d)) for d in fp)
        )

    text = re.sub(r"\b\d{1,3}(?:\.\d{3})+(?:,\d+)?\b", _grouped_num_repl, text)

    # Decimal comma (3,14)
    def _decimal_repl(m):
        ip, fp = m.group(1), m.group(2)
        return (
            _int_to_de(int(ip)) + " Komma " + " ".join(_int_to_de(int(d)) for d in fp)
        )

    text = re.sub(r"\b(\d+),(\d+)\b", _decimal_repl, text)

    # Plain integers
    text = re.sub(r"\b(\d+)\b", lambda m: _int_to_de(int(m.group(1))), text)

    # 10. Whitespace cleanup
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


# ── G2P class ────────────────────────────────────────────────────────────────


class DEG2P:
    """German G2P: normalize text then phonemize via espeak-ng."""

    def __init__(self):
        from .espeak import EspeakG2P

        self.espeak = EspeakG2P(language="de")

    def __call__(self, text) -> Tuple[str, None]:
        text = normalize_text_de(text)
        return self.espeak(text)
