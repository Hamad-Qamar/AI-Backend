"""
QueryPreprocessor — Module 1
Single responsibility: transform raw user input into a clean, normalised string.
Zero network calls, zero model imports.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STATIC DICTIONARIES
# ─────────────────────────────────────────────────────────────────────────────

# Construction-domain spell-correction dictionary.
# Keys are known misspellings / phonetic variants; values are canonical forms.
SPELL_CORRECTIONS: dict[str, str] = {
    # cement
    "cemnt": "cement", "cemment": "cement", "sement": "cement",
    "simont": "cement", "ciment": "cement", "cment": "cement",
    # bricks
    "briks": "bricks", "brik": "brick", "breeks": "bricks",
    "blicks": "bricks", "brics": "bricks",
    # tiles
    "tals": "tiles", "tils": "tiles", "tyls": "tiles", "tailes": "tiles",
    "tles": "tiles", "tile": "tile",
    # steel
    "stell": "steel", "stil": "steel", "steil": "steel", "stee": "steel",
    # sand
    "saand": "sand", "snad": "sand", "snd": "sand",
    # paint
    "pant": "paint", "paent": "paint", "paynt": "paint",
    # plumbing
    "plaming": "plumbing", "plumng": "plumbing", "plumbig": "plumbing",
    "plumbin": "plumbing",
    # electrical
    "electical": "electrical", "electrcal": "electrical",
    "electrial": "electrical", "electricl": "electrical",
    # flooring
    "florring": "flooring", "floring": "flooring", "flloring": "flooring",
    # foundation
    "foundtion": "foundation", "foundaton": "foundation",
    "founation": "foundation", "foudation": "foundation",
    # material
    "materail": "material", "materil": "material", "materal": "material",
    "materiall": "material", "matrail": "material",
    # construction
    "constuction": "construction", "constraction": "construction",
    "constructon": "construction", "costruciton": "construction",
    # marble
    "marbal": "marble", "marbel": "marble", "marbl": "marble",
    # granite
    "granit": "granite", "granate": "granite", "grannite": "granite",
    # roofing
    "roofng": "roofing", "rufing": "roofing", "roofin": "roofing",
    # bathroom / washroom
    "bathrom": "bathroom", "bathrom": "bathroom", "wasroom": "washroom",
    "washrom": "washroom",
    # kitchen
    "kichen": "kitchen", "kithen": "kitchen", "kitchn": "kitchen",
    # bedroom
    "bedrrom": "bedroom", "bedrom": "bedroom", "bedrm": "bedroom",
    # house
    "hose": "house", "houe": "house", "housse": "house",
    # building
    "bilding": "building", "bulding": "building", "biliding": "building",
    # estimate / estimation
    "estimte": "estimate", "estmate": "estimate", "estiamte": "estimate",
    "estimaton": "estimation",
    # recommendation
    "recommandation": "recommendation", "reccomendation": "recommendation",
    "recomendation": "recommendation",
    # quality
    "qualiy": "quality", "qualty": "quality", "qualit": "quality",
    # budget
    "budjet": "budget", "budgit": "budget", "budgt": "budget",
    # standard
    "standart": "standard", "standar": "standard",
    # premium
    "premum": "premium", "premiom": "premium",
    # economy
    "econmy": "economy", "economey": "economy",
    # labour
    "labur": "labour", "labouer": "labour", "laber": "labour",
    # category
    "catgory": "category", "categry": "category",
    # quantity
    "quantty": "quantity", "quantiy": "quantity",
    # seller
    "sellr": "seller", "seler": "seller",
    # listing
    "listng": "listing", "listin": "listing",
    # dashboard
    "dashbord": "dashboard", "dashborad": "dashboard",
    # delivery
    "delivry": "delivery", "delvery": "delivery",
    # payment
    "paymet": "payment", "paymnet": "payment",
    # order
    "ordr": "order", "orer": "order",
}


# Roman Urdu → English token translation table.
# Applied token-by-token to avoid partial-word collision.
ROMAN_URDU_MAP: dict[str, str] = {
    # materials
    "eent": "brick", "ent": "brick", "eentain": "bricks",
    "reti": "sand", "ritta": "sand", "baloo": "sand", "baaloo": "sand",
    "sarya": "steel", "sariya": "steel", "saria": "steel",
    "simant": "cement", "simont": "cement",
    "paani": "water", "pani": "water",
    "lakri": "wood", "lakkri": "wood", "timber": "wood",
    "sheesha": "glass", "sheesham": "sheesham wood",
    "rang": "paint", "rangai": "painting",
    "patthar": "stone", "pathar": "stone",
    "tamba": "copper", "loha": "iron", "faulad": "steel",
    "gitti": "gravel", "gravel": "gravel",
    "cheeni": "tiles", "tiles": "tiles",
    "marmar": "marble", "sangmarmar": "marble",
    # construction terms
    "bunyad": "foundation", "neev": "foundation",
    "chhat": "roof", "chat": "roof",
    "dewar": "wall", "diwar": "wall",
    "darwaza": "door", "darwazay": "doors",
    "khidki": "window", "khidkian": "windows",
    "zameen": "floor", "zamin": "floor",
    "chhat": "ceiling",
    "kache": "raw", "pukka": "finished",
    # cost / money
    "kharcha": "cost", "kharche": "cost",
    "kitna": "how much", "kitnay": "how much",
    "lakh": "lakh", "lac": "lakh",
    "sasta": "cheap", "sasti": "cheap", "sastaay": "cheap",
    "mehnga": "expensive", "mehnga": "expensive",
    "qeemat": "price", "daam": "price", "rate": "rate",
    "hisaab": "calculation", "andaza": "estimate",
    "bajat": "budget", "bujat": "budget",
    # sizes
    "marla": "marla", "kanal": "kanal",
    "gaz": "yard", "foot": "foot", "fit": "foot",
    "inch": "inch",
    # rooms
    "kamra": "room", "kamray": "rooms",
    "ghuslkhana": "bathroom", "gusalkkhana": "bathroom",
    "baithak": "lounge",
    "rasoi": "kitchen", "rasoiee": "kitchen",
    # actions
    "chahiye": "need", "chahiay": "need",
    "bnao": "build", "banao": "build", "banana": "build",
    "dikhao": "show", "dikhatay": "show",
    "batao": "tell", "bataein": "tell",
    "lagao": "install", "lagana": "install",
    "karo": "do", "karna": "do",
    "kaise": "how", "kaisay": "how",
    "kahan": "where", "kahan": "where",
    "hai": "", "hain": "", "ka": "of", "ki": "of", "ke": "of",
    "mujhe": "i need", "mujhay": "i need",
    "mera": "my", "meri": "my",
    "liye": "for", "liay": "for",
    "aur": "and", "ya": "or",
    "nahi": "not", "nahin": "not",
    "wala": "", "wali": "", "walay": "",
    # roles
    "kharidaar": "buyer", "baichne": "seller",
    "thekedar": "contractor",
}


# Short query expansion patterns.
# If the cleaned query has <=4 tokens AND matches a key, append the value.
SHORT_QUERY_EXPANSIONS: dict[str, str] = {
    "marla": "house construction materials",
    "kanal": "house construction materials",
    "sqft":  "construction materials",
    "5 marla": "house construction materials estimate",
    "10 marla": "house construction materials estimate",
    "cement": "types uses recommendation",
    "tiles": "flooring recommendation types",
    "steel": "construction steel recommendation",
    "bricks": "construction bricks quantity",
    "marble": "flooring marble types recommendation",
    "plumbing": "plumbing materials list",
    "electrical": "electrical materials wiring",
    "paint": "house paint types recommendation",
    "cost": "construction project cost estimation",
    "budget": "construction budget estimation",
}


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class QueryPreprocessor:
    """
    Transform raw user input into a clean, normalised string.
    Processing order:
      1. Whitespace & punctuation normalisation
      2. Spell correction (construction-domain only)
      3. Roman Urdu token translation
      4. Short-query expansion
    """

    def clean(self, raw: str) -> str:
        """
        Full preprocessing pipeline.
        Always returns a non-empty string (minimum "[empty query]" sentinel).
        """
        if not raw or not raw.strip():
            return "[empty query]"

        text = self._normalise_whitespace(raw)
        text = self._spell_correct(text)
        text = self._translate_roman_urdu(text)
        text = self._expand_short_query(text)

        return text.strip() or "[empty query]"

    def detect_language_hint(self, raw: str) -> str:
        """
        Classify the language character of the raw query.
        Returns: "roman_urdu" | "mixed" | "english"
        Does NOT affect preprocessing — purely for metadata annotation.
        """
        if not raw or not raw.strip():
            return "english"

        tokens = raw.lower().split()
        urdu_hits = sum(1 for t in tokens if t in ROMAN_URDU_MAP)
        ratio = urdu_hits / max(len(tokens), 1)

        if ratio >= 0.6:
            return "roman_urdu"
        elif ratio >= 0.2:
            return "mixed"
        return "english"

    # ── private helpers ───────────────────────────────────────────────────────

    def _normalise_whitespace(self, text: str) -> str:
        """Lowercase, strip, collapse whitespace, remove junk punctuation."""
        text = text.lower().strip()
        # Preserve hyphens and apostrophes as they appear in material names
        text = re.sub(r"[^\w\s\-']", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _spell_correct(self, text: str) -> str:
        """
        Token-by-token spell correction against the static dictionary.
        Only corrects if the token exactly matches a known misspelling.
        Does NOT attempt to guess general vocabulary.
        """
        tokens = text.split()
        corrected = [SPELL_CORRECTIONS.get(t, t) for t in tokens]
        return " ".join(corrected)

    def _translate_roman_urdu(self, text: str) -> str:
        """
        Token-by-token Roman Urdu → English translation.
        Tokens that map to empty string (stop-word equivalents) are dropped.
        """
        tokens = text.split()
        translated = []
        for token in tokens:
            translation = ROMAN_URDU_MAP.get(token)
            if translation is None:
                # Not a Roman Urdu word — keep as-is
                translated.append(token)
            elif translation:
                # Has a meaningful English equivalent
                translated.append(translation)
            # else: maps to "" — drop token (grammatical filler)
        return " ".join(translated)

    def _expand_short_query(self, text: str) -> str:
        """
        If the query is ≤4 tokens, check if it matches a known short-form pattern
        and append expansion context tokens.
        """
        tokens = text.split()
        if len(tokens) > 4:
            return text

        # Try multi-token key first (e.g. "5 marla"), then single token
        for n in (2, 1):
            prefix = " ".join(tokens[:n])
            if prefix in SHORT_QUERY_EXPANSIONS:
                expansion = SHORT_QUERY_EXPANSIONS[prefix]
                # Only append if expansion words are not already in the text
                new_words = [w for w in expansion.split() if w not in tokens]
                return text + " " + " ".join(new_words) if new_words else text

        return text
