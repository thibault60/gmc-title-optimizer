import os
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

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

STOPWORDS_FR = {"de", "la", "le", "les", "des", "du", "d", "et", "a", "à", "en", "pour", "avec", "sur", "dans"}

# =========================
# DATA STRUCTURES
# =========================
@dataclass
class KeywordRow:
    keyword: str
    volume: float
    ctr: float

# =========================
# UTILITIES
# =========================
def read_any(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

def as_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        # gère "1 234" et "12,3"
        s = str(x).replace(" ", "").replace(",", ".")
        m = re.search(r"-?\d+(\.\d+)?", s)
        return float(m.group(0)) if m else default
    except Exception:
        return default

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
    w_relevance: float = 0.60,
    w_volume: float = 0.30,
    w_ctr: float = 0.10,
    top_k: int = 6,
) -> List[Tuple[KeywordRow, float, float]]:
    vols = _minmax([k.volume for k in keywords])
    ctrs = _minmax([k.ctr for k in keywords])

    scored = []
    p = product_text.lower()
    for k, v_norm, c_norm in zip(keywords, vols, ctrs):
        rel = fuzz.token_set_ratio(p, k.keyword.lower()) / 100.0
        score = (w_relevance * rel) + (w_volume * v_norm) + (w_ctr * c_norm)
        scored.append((k, score, rel))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

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

def enforce_basic_rules(title: str, brand: str) -> str:
    t = (title or "").strip()
    if t and not t.lower().startswith("nappe"):
        t = "Nappe " + t
    if brand.lower() not in t.lower():
        t = f"{t} - {brand}"
    if len(t) > MAX_TITLE_LEN:
        t = t[:MAX_TITLE_LEN].rstrip()
    return t

def added_terms(old: str, new: str) -> List[str]:
    def tok(s: str) -> set:
        s = re.sub(r"[^\w\s\-×x]", " ", (s or "").lower())
        return {w for w in s.split() if len(w) > 2 and w not in STOPWORDS_FR}
    o, n = tok(old), tok(new)
    add = sorted(list(n - o))
    return add[:12]

# =========================
# OPENAI CLIENT (Responses + JSON Schema)
# =========================
def extract_output_text(resp_json: Dict[str, Any]) -> str:
    for item in resp_json.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                    return c["text"]
    raise ValueError("Impossible d'extraire output_text.")

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
        raise RuntimeError(f"OpenAI API error {r.status_code}: {r.text}")

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
    product_text = f"{current_title}\n{description}".strip()
    ranked = rank_keywords_for_product(product_text, keywords, top_k=6)

    candidates = [
        {"keyword": k.keyword, "volume": k.volume, "ctr": k.ctr, "relevance": round(rel, 3), "score": round(score, 3)}
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

Mots-clés candidats (Volume/CTR + score de pertinence):
{candidates}

Contraintes strictes:
- Le title DOIT commencer par "Nappe".
- Utiliser 1 à 2 mots-clés parmi les candidats SI (et seulement si) cohérents avec le produit décrit.
- N'ajoute PAS d'attributs non présents dans le title actuel ou la description (ne rien inventer).
- Pas de promo, pas de prix, pas de superlatifs gratuits.
- Marque une seule fois, de préférence en fin (ex: "- {brand}").
- Longueur cible 70–150 caractères.

Rends:
- optimized_title
- used_keywords (0 à 2)
- notes (1 phrase: pourquoi ces mots-clés)
""".strip()

    out = openai_generate_structured(
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=schema,
        temperature=0.2,
        max_output_tokens=240,
    )

    opt = enforce_basic_rules(out["optimized_title"], brand)
    issues = validate_title(opt, brand)
    adds = added_terms(current_title, opt)

    return {
        "title_current": current_title,
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
st.title("Optimisation Titles GMC — Verticale Nappes (Garnier-Thiebaut)")

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
    model = st.selectbox("Modèle", ["gpt-5-mini", "gpt-4o-mini"], index=0)
    n_rows = st.slider("Nombre de lignes à traiter (démo)", 1, 200, 10)
    st.caption("Astuce: commence par 10, puis augmente.")

st.subheader("1) Import des données")
prod_file = st.file_uploader("Produits (CSV/XLSX) — colonnes: title, description", type=["csv", "xlsx"])
kw_file = st.file_uploader("Mots-clés Nappes (CSV/XLSX) — colonnes: keyword, volume, ctr", type=["csv", "xlsx"])

if not prod_file or not kw_file:
    st.info("Importe les 2 fichiers pour lancer l’optimisation.")
    st.stop()

df_prod = read_any(prod_file)
df_kw = read_any(kw_file)

st.write("Aperçu produits", df_prod.head())
st.write("Aperçu mots-clés", df_kw.head())

st.subheader("2) Mapping colonnes")
col_title = st.selectbox("Colonne Title (produits)", df_prod.columns, index=list(df_prod.columns).index("title") if "title" in df_prod.columns else 0)
col_desc = st.selectbox("Colonne Description (produits)", df_prod.columns, index=list(df_prod.columns).index("description") if "description" in df_prod.columns else 0)

col_kw = st.selectbox("Colonne Keyword", df_kw.columns, index=list(df_kw.columns).index("keyword") if "keyword" in df_kw.columns else 0)
col_vol = st.selectbox("Colonne Volume", df_kw.columns, index=list(df_kw.columns).index("volume") if "volume" in df_kw.columns else 0)
col_ctr = st.selectbox("Colonne CTR", df_kw.columns, index=list(df_kw.columns).index("ctr") if "ctr" in df_kw.columns else 0)

keywords: List[KeywordRow] = []
for _, r in df_kw.iterrows():
    k = str(r[col_kw]).strip()
    if not k or k.lower() == "nan":
        continue
    keywords.append(
        KeywordRow(
            keyword=k,
            volume=as_float(r[col_vol], 0.0),
            ctr=as_float(r[col_ctr], 0.0),
        )
    )

st.subheader("3) Optimisation")
if st.button("Optimiser les titles (Nappes)"):
    df = df_prod.head(n_rows).copy()

    results = []
    prog = st.progress(0)

    for idx, row in df.iterrows():
        cur_title = str(row[col_title]) if pd.notna(row[col_title]) else ""
        desc = str(row[col_desc]) if pd.notna(row[col_desc]) else ""

        out = optimize_nappe_title(
            api_key=api_key,
            model=model,
            brand=brand,
            current_title=cur_title,
            description=desc,
            keywords=keywords,
        )
        results.append(out)
        prog.progress(min(1.0, len(results) / max(1, n_rows)))

    res_df = pd.DataFrame(results)
    st.success("Terminé ✅")
    st.dataframe(res_df, use_container_width=True)

    csv = res_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger le résultat (CSV)",
        data=csv,
        file_name="gmc_titles_nappes_optimized.csv",
        mime="text/csv",
    )
