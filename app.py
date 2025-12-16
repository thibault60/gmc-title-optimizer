import os
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from io import BytesIO

import requests
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz

# =========================
# CONFIG
# =========================
OPENAI_URL = "https://api.openai.com/v1/responses"
BRAND_DEFAULT = "Garnier-Thiebaut"
VERTICAL_DEFAULT = "Nappes"
MAX_TITLE_LEN = 150

STOPWORDS_FR = {
    "de", "la", "le", "les", "des", "du", "d", "et", "a", "à", "en",
    "pour", "avec", "sur", "dans", "au", "aux", "un", "une", "the", "of", "in"
}

# =========================
# DATA STRUCTURES
# =========================
@dataclass
class KeywordRow:
    keyword: str
    volume: float

# =========================
# FILE READERS
# =========================
def read_any(file) -> pd.DataFrame:
    """
    Supporte CSV et XLSX.
    - CSV: essaie ',' puis ';'
    - XLSX: nécessite openpyxl
    """
    name = file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(file)
        except Exception:
            file.seek(0)
            return pd.read_csv(file, sep=";")
    return pd.read_excel(file, engine="openpyxl")

def as_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        s = str(x).replace(" ", "").replace(",", ".")
        m = re.search(r"-?\d+(\.\d+)?", s)
        return float(m.group(0)) if m else default
    except Exception:
        return default

def df_to_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "nappes") -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()

# =========================
# TEXT NORMALIZATION
# =========================
def strip_accents(s: str) -> str:
    if not s:
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

def has_made_in_france(description: str) -> bool:
    """
    Détecte 'Fabriqué en France' même avec accents/typos, + quelques variantes.
    """
    d = strip_accents((description or "")).lower()
    d = re.sub(r"\s+", " ", d).strip()

    patterns = [
        r"fabrique en france",
        r"fabriquee en france",
        r"fabriqu[eé] en france",  # au cas où accents restent
        r"made in france",
        r"fabrication francaise",
        r"fabrique a?u?x? france",  # tolérance ultra large
    ]
    return any(re.search(p, d) for p in patterns)

# =========================
# KEYWORD SCORING
# =========================
def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if math.isclose(vmin, vmax):
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]

def rank_keywords_for_product(
    product_text: str,
    keywords: List[KeywordRow],
    *,
    w_relevance: float = 0.70,
    w_volume: float = 0.30,
    top_k: int = 6,
) -> List[Tuple[KeywordRow, float, float]]:
    vols = _minmax([k.volume for k in keywords])

    scored = []
    p = (product_text or "").lower()
    for k, v_norm in zip(keywords, vols):
        rel = fuzz.token_set_ratio(p, k.keyword.lower()) / 100.0
        score = (w_relevance * rel) + (w_volume * v_norm)
        scored.append((k, score, rel))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

# =========================
# TITLE FORMATTING / QA
# =========================
def normalize_spaces(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    s = re.sub(r"\s*-\s*", " - ", s)  # uniformise séparateurs
    return s

def smart_title_case(title: str, brand: str) -> str:
    """
    "Camel Case" version e-commerce : majuscule initiale des mots importants.
    Conserve dimensions, % et la marque.
    """
    t = normalize_spaces(title)

    # Protéger marque
    placeholder = "__BRAND__"
    t_protected = t.replace(brand, placeholder)

    def transform_token(tok: str) -> str:
        if not tok:
            return tok

        low = tok.lower()

        # keep common measurement patterns / numbers
        if re.match(r"^\d+([x×]\d+)+$", tok.lower()):
            return tok
        if re.match(r"^\d+(cm|mm|m)$", low):
            return low  # "cm" plutôt en minuscule
        if re.match(r"^\d+(\.\d+)?%$", tok):
            return tok
        if low in STOPWORDS_FR:
            return low

        # mots avec apostrophe : d' / l'
        if re.match(r"^[dl]'[a-zàâçéèêëîïôûùüÿñæœ].+", tok.lower()):
            prefix, rest = tok.split("'", 1)
            return prefix.lower() + "'" + (rest[:1].upper() + rest[1:].lower())

        # hyphen words : "anti-tache", "garnier-thiebaut" etc.
        if "-" in tok:
            parts = tok.split("-")
            parts2 = []
            for p in parts:
                if not p:
                    parts2.append(p)
                else:
                    lp = p.lower()
                    if lp in STOPWORDS_FR:
                        parts2.append(lp)
                    else:
                        parts2.append(lp[:1].upper() + lp[1:])
            return "-".join(parts2)

        return low[:1].upper() + low[1:]

    tokens = t_protected.split(" ")
    # premier mot toujours capitalisé
    out = []
    for i, tok in enumerate(tokens):
        if i == 0:
            out.append(tok[:1].upper() + tok[1:].lower() if tok else tok)
        else:
            out.append(transform_token(tok))

    t2 = " ".join(out).replace(placeholder, brand)

    # Phrase fixe exigée
    t2 = re.sub(r"\bMade In France\b", "Made in France", t2)
    return normalize_spaces(t2)

def truncate_preserve_suffix(main: str, suffix: str, max_len: int) -> str:
    """
    Tronque main pour que (main + suffix) <= max_len, en préservant suffix.
    """
    main = (main or "").rstrip()
    suffix = (suffix or "").lstrip()

    if not suffix:
        return main[:max_len].rstrip()

    if len(suffix) >= max_len:
        return suffix[:max_len].rstrip()

    allowed = max_len - len(suffix)
    main_cut = main[:allowed].rstrip()
    return (main_cut + suffix).rstrip()

def ensure_brand_suffix(title: str, brand: str) -> str:
    """
    Assure que le title termine par " - {brand}" et que la marque n'apparait qu'une fois.
    """
    t = normalize_spaces(title)

    # Supprime occurrences multiples de la marque (simple)
    # On retire toutes occurrences puis on la remet en fin
    t_wo_brand = re.sub(re.escape(brand), "", t, flags=re.IGNORECASE)
    t_wo_brand = normalize_spaces(t_wo_brand).strip(" -")

    suffix = f" - {brand}"
    if not t_wo_brand:
        return f"{brand}"[:MAX_TITLE_LEN]

    return (t_wo_brand + suffix).strip()

def add_made_in_france_if_needed(title: str, brand: str, should_add: bool) -> str:
    """
    Ajoute 'Made in France' avant la marque si should_add=True.
    Gère déduplication et conservation fin de title.
    """
    t = normalize_spaces(title)

    # retire toute occurrence existante
    t = re.sub(r"\bmade\s+in\s+france\b", "", t, flags=re.IGNORECASE)
    t = normalize_spaces(t).strip(" -")

    # assure marque en fin d'abord
    t = ensure_brand_suffix(t, brand)

    if not should_add:
        return t

    suffix_brand = f" - {brand}"
    if t.endswith(suffix_brand):
        main = t[:-len(suffix_brand)].rstrip()
        # évite double tirets
        main = main.strip(" -")
        new_suffix = f" - Made in France - {brand}"
        # tronque en préservant suffix
        return truncate_preserve_suffix(main, new_suffix, MAX_TITLE_LEN)

    # fallback
    t2 = f"{t} - Made in France"
    return truncate_preserve_suffix(t2, "", MAX_TITLE_LEN)

def enforce_basic_rules(title: str, brand: str, made_in_france: bool) -> str:
    t = normalize_spaces(title)

    # doit commencer par "Nappe"
    if t and not t.lower().startswith("nappe"):
        t = "Nappe " + t

    # marque en fin + made in france
    t = add_made_in_france_if_needed(t, brand, made_in_france)

    # limite 150 (préservée via truncate_preserve_suffix dans made_in_france path, mais re-sécurise)
    t = t[:MAX_TITLE_LEN].rstrip()

    # title case (camel-ish)
    t = smart_title_case(t, brand)

    # re-sécurise longueur après casing
    t = t[:MAX_TITLE_LEN].rstrip()
    return t

def validate_title(title: str, brand: str) -> List[str]:
    issues = []
    t = (title or "").strip()
    if not t:
        return ["title vide"]
    if len(t) > MAX_TITLE_LEN:
        issues.append(f"trop long ({len(t)} > {MAX_TITLE_LEN})")
    if not t.lower().startswith("nappe"):
        issues.append("ne commence pas par 'Nappe'")
    count_brand = t.lower().count(brand.lower())
    if count_brand == 0:
        issues.append("marque absente")
    elif count_brand > 1:
        issues.append("marque présente plusieurs fois")
    if re.search(r"\bpromo\b|\bréduc\b|\breduction\b|%|€", t.lower()):
        issues.append("contient promo/prix")
    return issues

def added_terms(old: str, new: str) -> List[str]:
    def tok(s: str) -> set:
        s = re.sub(r"[^\w\s\-×x]", " ", (s or "").lower())
        return {w for w in s.split() if len(w) > 2 and w not in STOPWORDS_FR}
    o, n = tok(old), tok(new)
    add = sorted(list(n - o))
    return add[:12]

# =========================
# OPENAI HELPERS
# =========================
def extract_output_text(resp_json: Dict[str, Any]) -> str:
    for item in resp_json.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                    return c["text"]
    raise ValueError("Impossible d'extraire output_text.")

def openai_list_models(api_key: str) -> List[str]:
    r = requests.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    if r.status_code != 200:
        try:
            j = r.json()
            msg = j.get("error", {}).get("message", r.text)
        except Exception:
            msg = r.text
        raise RuntimeError(f"List models failed HTTP {r.status_code}: {msg}")

    j = r.json()
    return sorted([m["id"] for m in j.get("data", []) if "id" in m])

def openai_generate_structured(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: Dict[str, Any],
    temperature: float = 0.2,
    max_output_tokens: int = 240,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "gmc_title_optimization",
                "strict": True,
                "schema": schema,
            }
        },
    }

    r = requests.post(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=60,
    )

    if r.status_code < 200 or r.status_code >= 300:
        try:
            j = r.json()
            err = j.get("error", {})
            msg = err.get("message", r.text)
            code = err.get("code", "")
            typ = err.get("type", "")
            raise RuntimeError(f"OpenAI HTTP {r.status_code} | type={typ} code={code} | {msg}")
        except Exception:
            raise RuntimeError(f"OpenAI HTTP {r.status_code} | {r.text}")

    data = r.json()
    txt = extract_output_text(data)
    return json.loads(txt)

# =========================
# OPTIMIZER (NAPPES)
# =========================
def optimize_nappe_title(
    *,
    api_key: str,
    model: str,
    brand: str,
    current_title: str,
    description: str,
    keywords: List[KeywordRow],
) -> Dict[str, Any]:
    current_title = (current_title or "")[:220]
    description = (description or "")[:1200]

    made_in_fr = has_made_in_france(description)

    product_text = f"{current_title}\n{description}".strip()
    ranked = rank_keywords_for_product(product_text, keywords, top_k=6)

    candidates = [
        {"keyword": k.keyword, "volume": k.volume, "relevance": round(rel, 3), "score": round(score, 3)}
        for (k, score, rel) in ranked
    ]

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "optimized_title": {"type": "string", "minLength": 10, "maxLength": 170},
            "used_keywords": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 2},
            "notes": {"type": "string"},
        },
        "required": ["optimized_title", "used_keywords", "notes"],
    }

    system_prompt = (
        "Tu es expert SEO e-commerce + Google Merchant Center FR. "
        "Tu optimises des titles Shopping pour une marque premium."
    )

    user_prompt = f"""
Verticale: {VERTICAL_DEFAULT}
Marque: {brand}

Title actuel:
{current_title}

Description:
{description}

Mots-clés candidats (Volume + score de pertinence):
{candidates}

Contraintes strictes:
- Le title DOIT commencer par "Nappe".
- Utiliser 1 à 2 mots-clés parmi les candidats SI (et seulement si) cohérents avec le produit décrit.
- N'ajoute PAS d'attributs non présents dans le title actuel ou la description (ne rien inventer).
- Pas de promo, pas de prix, pas de superlatifs gratuits.
- Marque une seule fois, de préférence en fin (ex: "- {brand}").
- Longueur max: {MAX_TITLE_LEN} caractères.
""".strip()

    try:
        out = openai_generate_structured(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            temperature=0.2,
            max_output_tokens=240,
        )
    except RuntimeError as e:
        msg = str(e).lower()
        if ("model" in msg and "not found" in msg) or ("model_not_found" in msg):
            out = openai_generate_structured(
                api_key=api_key,
                model="gpt-4o-mini",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=schema,
                temperature=0.2,
                max_output_tokens=240,
            )
        else:
            raise

    opt = enforce_basic_rules(out["optimized_title"], brand, made_in_fr)
    issues = validate_title(opt, brand)
    adds = added_terms(current_title, opt)

    return {
        "title_current": current_title,
        "description_has_fabrique_france": "Oui" if made_in_fr else "Non",
        "title_optimized": opt,
        "added_terms": " | ".join(adds),
        "used_keywords": " | ".join(out.get("used_keywords", [])),
        "notes": out.get("notes", ""),
        "validation_issues": " | ".join(issues),
        "keyword_candidates": candidates,
    }

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="GMC Title Optimizer (Nappes)", layout="wide")
st.title("Optimisation Titles GMC — Nappes (keyword + volume) → export Excel")

# secrets -> env
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

api_key = os.environ.get("OPENAI_API_KEY", "").strip()
if not api_key:
    st.warning("Ajoute OPENAI_API_KEY dans .streamlit/secrets.toml ou en variable d’environnement.")
    st.stop()

with st.sidebar:
    st.header("Paramètres")
    brand = st.text_input("Marque", BRAND_DEFAULT)
    model = st.selectbox("Modèle", ["gpt-4o-mini", "gpt-5-mini"], index=0)
    n_rows = st.slider("Nombre de lignes à traiter (démo)", 1, 200, 10)

    if st.button("Tester la clé + lister les modèles"):
        try:
            models = openai_list_models(api_key)
            st.success("OK : clé API valide ✅")
            pick = [m for m in models if ("gpt-4o" in m or "gpt-5" in m)]
            st.write("Modèles détectés (filtrés):")
            st.write(pick[:80] if pick else models[:80])
        except Exception as e:
            st.error(str(e))

st.subheader("1) Import des données")
prod_file = st.file_uploader(
    "Produits (CSV/XLSX) — colonnes: title, description",
    type=["csv", "xlsx"]
)
kw_file = st.file_uploader(
    "Mots-clés Nappes (CSV/XLSX) — colonnes: keyword, volume",
    type=["csv", "xlsx"]
)

if not prod_file or not kw_file:
    st.info("Importe les 2 fichiers pour lancer l’optimisation.")
    st.stop()

df_prod = read_any(prod_file)
df_kw = read_any(kw_file)

st.write("Aperçu produits", df_prod.head())
st.write("Aperçu mots-clés", df_kw.head())

st.subheader("2) Mapping colonnes")
col_title = st.selectbox(
    "Colonne Title (produits)",
    df_prod.columns,
    index=list(df_prod.columns).index("title") if "title" in df_prod.columns else 0
)
col_desc = st.selectbox(
    "Colonne Description (produits)",
    df_prod.columns,
    index=list(df_prod.columns).index("description") if "description" in df_prod.columns else 0
)

col_kw = st.selectbox(
    "Colonne Keyword (mots-clés)",
    df_kw.columns,
    index=list(df_kw.columns).index("keyword") if "keyword" in df_kw.columns else 0
)
col_vol = st.selectbox(
    "Colonne Volume (mots-clés)",
    df_kw.columns,
    index=list(df_kw.columns).index("volume") if "volume" in df_kw.columns else 0
)

keywords: List[KeywordRow] = []
for _, r in df_kw.iterrows():
    k = str(r[col_kw]).strip()
    if not k or k.lower() == "nan":
        continue
    keywords.append(KeywordRow(keyword=k, volume=as_float(r[col_vol], 0.0)))

if not keywords:
    st.error("Aucun mot-clé valide détecté (vérifie les colonnes keyword/volume).")
    st.stop()

st.subheader("3) Optimisation + Export Excel")
if st.button("Optimiser les titles (Nappes)"):
    df = df_prod.head(n_rows).copy()

    results = []
    prog = st.progress(0)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        cur_title = str(row[col_title]) if pd.notna(row[col_title]) else ""
        desc = str(row[col_desc]) if pd.notna(row[col_desc]) else ""

        try:
            out = optimize_nappe_title(
                api_key=api_key,
                model=model,
                brand=brand,
                current_title=cur_title,
                description=desc,
                keywords=keywords,
            )
            results.append(out)
        except Exception as e:
            results.append({
                "title_current": cur_title,
                "description_has_fabrique_france": "",
                "title_optimized": "",
                "added_terms": "",
                "used_keywords": "",
                "notes": "",
                "validation_issues": f"ERREUR: {str(e)}",
                "keyword_candidates": "",
            })

        prog.progress(min(1.0, i / max(1, n_rows)))

    res_df = pd.DataFrame(results)
    st.success("Terminé ✅")
    st.dataframe(res_df, use_container_width=True)

    # Export XLSX
    xlsx_bytes = df_to_xlsx_bytes(res_df, sheet_name="nappes")
    st.download_button(
        "Télécharger le résultat (Excel .xlsx)",
        data=xlsx_bytes,
        file_name="gmc_titles_nappes_optimized.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
