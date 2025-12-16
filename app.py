import os
import json
import math
import re
import time
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
    "de","la","le","les","des","du","d","et","a","à","en","pour","avec","sur","dans",
    "au","aux","un","une","the","of","in"
}

# =========================
# DATA STRUCTURES
# =========================
@dataclass
class KeywordRow:
    keyword: str
    volume: float

# =========================
# FILE READERS / WRITERS
# =========================
def read_any(file) -> pd.DataFrame:
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

def normalize_separators(s: str) -> str:
    """
    Normalise uniquement les séparateurs entre blocs (ex: " - "), SANS casser les tirets internes.
    -> "Garnier-Thiebaut" reste "Garnier-Thiebaut"
    """
    if not s:
        return ""
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip()

    # Convertit des séparateurs alternatifs en " - "
    s = re.sub(r"\s*[|,]\s*", " - ", s)

    # Normalise uniquement les tirets avec espaces autour: " a - b " => " a - b "
    s = re.sub(r"\s+-\s+", " - ", s)

    # Collapsing des répétitions " -  -  - "
    s = re.sub(r"(?:\s-\s){2,}", " - ", s)

    # Nettoie " - " en début/fin
    s = s.strip()
    s = re.sub(r"^(?:-\s*)+", "", s).strip()
    s = re.sub(r"(?:\s*-\s*)+$", "", s).strip()

    return s

def has_made_in_france(description: str) -> bool:
    d = strip_accents((description or "")).lower()
    d = re.sub(r"\s+", " ", d).strip()
    return (
        "fabrique en france" in d
        or "fabriquee en france" in d
        or "made in france" in d
        or "fabrication francaise" in d
    )

def remove_made_in_france(text: str) -> str:
    t = re.sub(r"\bmade\s+in\s+france\b", " ", text or "", flags=re.IGNORECASE)
    return normalize_separators(t).strip(" -")

def brand_variant_regex(brand: str) -> re.Pattern:
    """
    Match variantes:
    - Garnier-Thiebaut
    - Garnier - Thiebaut
    - Garnier Thiebaut
    - Garnier–Thiebaut
    """
    b = strip_accents(brand).lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", b) if t]
    if not tokens:
        return re.compile(re.escape(brand), re.IGNORECASE)

    joiner = r"(?:\s*[-–—]\s*|\s+)"  # hyphen avec/sans espaces, ou espace
    pattern = r"\b" + joiner.join(map(re.escape, tokens)) + r"\b"
    return re.compile(pattern, re.IGNORECASE)

def remove_brand_occurrences(text: str, brand: str) -> str:
    """
    Retire TOUTES les occurrences de la marque (toutes variantes), puis nettoie les séparateurs.
    """
    t = normalize_separators(text)
    pat = brand_variant_regex(brand)
    t = pat.sub(" ", t)
    t = normalize_separators(t)
    t = t.strip(" -|,")
    return t

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
# TITLE CASE ("Camel case visuel")
# =========================
def smart_title_case(title: str, brand: str) -> str:
    """
    Nappe Coton Mélangé Antitache...
    - conserve dimensions, unités, acronymes
    - conserve la marque (format canonical) inchangée
    - force "Made in France" (in minuscule)
    """
    t = normalize_separators(title)

    placeholder = "__BRAND__"
    t = t.replace(brand, placeholder)

    def is_dimension(tok: str) -> bool:
        return bool(re.fullmatch(r"\d+([x×]\d+)+", tok.lower()))

    def is_measure(tok: str) -> bool:
        return bool(re.fullmatch(r"\d+(cm|mm|m)", tok.lower()))

    def is_percent(tok: str) -> bool:
        return bool(re.fullmatch(r"\d+(\.\d+)?%", tok))

    def is_acronym(tok: str) -> bool:
        return bool(re.fullmatch(r"[A-Z0-9]{2,}([\-][A-Z0-9]{2,})*", tok))

    def cap_word(w: str) -> str:
        return w[:1].upper() + w[1:].lower() if w else w

    def transform_token(tok: str, is_first: bool) -> str:
        if not tok:
            return tok
        if tok == placeholder:
            return placeholder

        if is_dimension(tok) or is_measure(tok) or is_percent(tok) or is_acronym(tok):
            return tok

        low = tok.lower()

        if (low in STOPWORDS_FR) and not is_first:
            return low

        # d'/l'
        if re.fullmatch(r"[dl]'[a-zàâçéèêëîïôûùüÿñæœ].+", low):
            prefix, rest = tok.split("'", 1)
            return prefix.lower() + "'" + cap_word(rest)

        # mot composé (ANTI-TACHE => Anti-Tache)
        if "-" in tok:
            parts = tok.split("-")
            outp = []
            for p in parts:
                if not p:
                    outp.append(p)
                    continue
                lp = p.lower()
                if (lp in STOPWORDS_FR) and not is_first:
                    outp.append(lp)
                else:
                    outp.append(cap_word(p))
            return "-".join(outp)

        return cap_word(tok)

    tokens = t.split(" ")
    out = [transform_token(tok, is_first=(i == 0)) for i, tok in enumerate(tokens)]
    t2 = " ".join(out)

    t2 = re.sub(r"\bMade\s+In\s+France\b", "Made in France", t2)
    t2 = re.sub(r"\bMade\s+in\s+France\b", "Made in France", t2)

    t2 = t2.replace(placeholder, brand)
    return normalize_separators(t2)

# =========================
# FINAL ENFORCEMENT (brand unique fin)
# =========================
def truncate_preserve_suffix(main: str, suffix: str, max_len: int) -> str:
    main = (main or "").rstrip()
    suffix = (suffix or "").lstrip()

    if not suffix:
        return main[:max_len].rstrip()

    if len(suffix) >= max_len:
        return suffix[:max_len].rstrip()

    allowed = max_len - len(suffix)
    return (main[:allowed].rstrip() + suffix).rstrip()

def enforce_basic_rules(title: str, brand: str, made_in_france: bool, max_len: int = MAX_TITLE_LEN) -> str:
    """
    GARANTIES:
    - Marque supprimée partout (toutes variantes)
    - Made in France supprimé puis ré-ajouté si nécessaire
    - Préfixe "Nappe"
    - Suffix final:
        - " - Made in France - Garnier-Thiebaut" si détecté
        - sinon " - Garnier-Thiebaut"
    - Longueur <= 150 en préservant le suffix
    - Title case final
    """
    t = normalize_separators(title)

    # purge marque + made in france partout dans le corps
    t = remove_brand_occurrences(t, brand)
    t = remove_made_in_france(t)

    if t and not t.lower().startswith("nappe"):
        t = "Nappe " + t
    t = normalize_separators(t).strip(" -")

    suffix = f" - Made in France - {brand}" if made_in_france else f" - {brand}"
    t = truncate_preserve_suffix(t, suffix, max_len)
    t = smart_title_case(t, brand)

    # hard final (au cas où le casing / nettoyage réintroduit des variantes)
    t_main = remove_brand_occurrences(t, brand)
    t_main = remove_made_in_france(t_main)
    if t_main and not t_main.lower().startswith("nappe"):
        t_main = "Nappe " + t_main
    t_main = normalize_separators(t_main).strip(" -")
    t = truncate_preserve_suffix(t_main, suffix, max_len)
    t = smart_title_case(t, brand)
    t = t[:max_len].rstrip()

    return t

def validate_title(title: str, brand: str) -> List[str]:
    issues = []
    t = normalize_separators(title).strip()

    if not t:
        return ["title vide"]
    if len(t) > MAX_TITLE_LEN:
        issues.append(f"trop long ({len(t)} > {MAX_TITLE_LEN})")
    if not t.lower().startswith("nappe"):
        issues.append("ne commence pas par 'Nappe'")

    pat = brand_variant_regex(brand)
    occ = list(pat.finditer(t))
    if len(occ) != 1:
        issues.append("marque != 1 occurrence (variante incluse)")
    if not t.endswith(brand):
        issues.append("marque pas au format canonical en fin (doit finir par 'Garnier-Thiebaut')")

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
    retries: int = 3,
    backoff_s: float = 1.5,
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

    last_err = None
    for attempt in range(retries + 1):
        r = requests.post(
            OPENAI_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=90,
        )

        if 200 <= r.status_code < 300:
            data = r.json()
            txt = extract_output_text(data)
            return json.loads(txt)

        try:
            j = r.json()
            err = j.get("error", {})
            msg = err.get("message", r.text)
            code = err.get("code", "")
            typ = err.get("type", "")
            last_err = f"OpenAI HTTP {r.status_code} | type={typ} code={code} | {msg}"
        except Exception:
            last_err = f"OpenAI HTTP {r.status_code} | {r.text}"

        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            time.sleep(backoff_s * (attempt + 1))
            continue

        raise RuntimeError(last_err)

    raise RuntimeError(last_err or "OpenAI error (unknown)")

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
    per_call_sleep_s: float = 0.0,
) -> Dict[str, Any]:
    current_title = (current_title or "")[:220]
    description = (description or "")[:1500]

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
- Utiliser 1 à 2 mots-clés parmi les candidats SI cohérents avec le produit.
- N'ajoute PAS d'attributs non présents dans le title actuel ou la description.
- Pas de promo, pas de prix.
- Longueur max: {MAX_TITLE_LEN} caractères.
Rends:
- optimized_title
- used_keywords
- notes
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
            retries=3,
            backoff_s=1.5,
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
                retries=3,
                backoff_s=1.5,
            )
        else:
            raise

    opt = enforce_basic_rules(out["optimized_title"], brand, made_in_fr, max_len=MAX_TITLE_LEN)
    issues = validate_title(opt, brand)
    adds = added_terms(current_title, opt)

    if per_call_sleep_s and per_call_sleep_s > 0:
        time.sleep(per_call_sleep_s)

    return {
        "source_row_index": None,
        "title_current": current_title,
        "description_has_made_in_france": "Oui" if made_in_fr else "Non",
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

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

api_key = os.environ.get("OPENAI_API_KEY", "").strip()
if not api_key:
    st.warning("Ajoute OPENAI_API_KEY dans .streamlit/secrets.toml ou en variable d’environnement.")
    st.stop()

with st.sidebar:
    st.header("Paramètres")
    brand = st.text_input("Marque (canonical)", BRAND_DEFAULT)
    model = st.selectbox("Modèle", ["gpt-4o-mini", "gpt-5-mini"], index=0)
    per_call_sleep_s = st.slider("Pause entre appels API (sec)", 0.0, 2.0, 0.2, 0.1)
    st.caption("Augmente si tu as des 429 (rate limit).")

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
prod_file = st.file_uploader("Produits (CSV/XLSX) — colonnes: title, description", type=["csv", "xlsx"])
kw_file = st.file_uploader("Mots-clés Nappes (CSV/XLSX) — colonnes: keyword, volume", type=["csv", "xlsx"])

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

st.subheader("3) Traitement (batch / au-delà de la démo)")
total_rows = len(df_prod)
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    start_index = st.number_input("Index de départ (0 = 1ère ligne)", min_value=0, max_value=max(0, total_rows - 1), value=0, step=1)
with c2:
    treat_all = st.checkbox("Traiter toutes les lignes (à partir de l’index)", value=False)
with c3:
    if treat_all:
        n_rows = total_rows - int(start_index)
        st.write(f"Traitement prévu : **{n_rows}** ligne(s) (sur {total_rows})")
    else:
        n_rows = st.number_input("Nombre de lignes à traiter", min_value=1, max_value=max(1, total_rows - int(start_index)), value=min(200, max(1, total_rows - int(start_index))), step=1)

if st.button("Optimiser les titles (Nappes)"):
    start_index = int(start_index)
    n_rows = int(n_rows)
    end_index = min(total_rows, start_index + n_rows)

    results = []
    prog = st.progress(0)
    status = st.empty()

    for pos, i in enumerate(range(start_index, end_index), start=1):
        row = df_prod.iloc[i]
        cur_title = str(row[col_title]) if pd.notna(row[col_title]) else ""
        desc = str(row[col_desc]) if pd.notna(row[col_desc]) else ""

        status.write(f"Ligne {i} ({pos}/{end_index-start_index}) …")

        try:
            out = optimize_nappe_title(
                api_key=api_key,
                model=model,
                brand=brand,
                current_title=cur_title,
                description=desc,
                keywords=keywords,
                per_call_sleep_s=per_call_sleep_s,
            )
            out["source_row_index"] = i
            results.append(out)
        except Exception as e:
            results.append({
                "source_row_index": i,
                "title_current": cur_title,
                "description_has_made_in_france": "",
                "title_optimized": "",
                "added_terms": "",
                "used_keywords": "",
                "notes": "",
                "validation_issues": f"ERREUR: {str(e)}",
                "keyword_candidates": "",
            })

        prog.progress(min(1.0, pos / max(1, (end_index - start_index))))

    res_df = pd.DataFrame(results)
    st.success("Terminé ✅")
    st.dataframe(res_df, use_container_width=True)

    xlsx_bytes = df_to_xlsx_bytes(res_df, sheet_name="nappes")
    st.download_button(
        "Télécharger le résultat (Excel .xlsx)",
        data=xlsx_bytes,
        file_name=f"gmc_titles_nappes_optimized_{start_index}_{end_index-1}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
