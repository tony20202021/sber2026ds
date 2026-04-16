"""
compute_gender_embeddings.py — gender-similarity embeddings для MCC и TR кодов.

Использует spaCy ru_core_news_md (300d Word2Vec).
Для каждого кода вычисляет:
  - male_sim, female_sim, gender_diff (= male_sim - female_sim)
Сохраняет в results/gender_embeddings/*.csv.
"""
import os
import gc
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

MALE_WORDS = [
    "мужчина", "мужской", "парень", "муж", "сын", "отец", "дедушка",
    "дядя", "брат", "юноша", "мальчик", "папа", "дед",
]
FEMALE_WORDS = [
    "женщина", "женский", "девушка", "жена", "дочь", "мать", "бабушка",
    "тётя", "сестра", "девочка", "дама", "мама", "бабуля",
]


def main():
    os.makedirs("results/gender_embeddings", exist_ok=True)

    # 1. Читаем ТОЛЬКО уникальные описания (не весь train.csv)
    print("Reading unique code descriptions (small sample)...", flush=True)
    chunks = pd.read_csv(
        "data/train.csv",
        usecols=["mcc_code", "mcc_code_desc", "tr_type", "tr_type_desc"],
        dtype={"mcc_code": "int16", "tr_type": "int16"},
        chunksize=50_000,
    )
    mcc_seen, tr_seen = {}, {}
    for chunk in chunks:
        for _, row in chunk[["mcc_code","mcc_code_desc"]].drop_duplicates().iterrows():
            if row["mcc_code"] not in mcc_seen:
                mcc_seen[row["mcc_code"]] = row["mcc_code_desc"]
        for _, row in chunk[["tr_type","tr_type_desc"]].drop_duplicates().iterrows():
            if row["tr_type"] not in tr_seen:
                tr_seen[row["tr_type"]] = row["tr_type_desc"]
        if len(mcc_seen) >= 200 and len(tr_seen) >= 80:
            break
    mcc_map = pd.DataFrame({"mcc_code": list(mcc_seen.keys()),
                             "mcc_code_desc": list(mcc_seen.values())})
    tr_map  = pd.DataFrame({"tr_type": list(tr_seen.keys()),
                             "tr_type_desc": list(tr_seen.values())})
    print(f"  MCC: {len(mcc_map)} codes, TR: {len(tr_map)} types", flush=True)
    del chunks
    gc.collect()

    # 2. Загружаем spaCy
    print("Loading spaCy ru_core_news_md...", flush=True)
    import spacy
    nlp = spacy.load("ru_core_news_md", disable=["parser", "ner", "tagger", "morphologizer", "senter"])
    print(f"  Vector dim: {nlp.vocab.vectors_length}", flush=True)

    def centroid(words):
        vecs = []
        for w in words:
            tok = nlp(w)
            if tok.has_vector:
                v = tok.vector.copy()
                norm = np.linalg.norm(v)
                if norm > 0:
                    vecs.append(v / norm)
        return np.mean(vecs, axis=0) if vecs else np.zeros(nlp.vocab.vectors_length)

    male_vec   = centroid(MALE_WORDS)
    female_vec = centroid(FEMALE_WORDS)
    sim = cosine_similarity([male_vec], [female_vec])[0][0]
    print(f"  cos(male, female) centroid: {sim:.3f}", flush=True)

    def embed_text(text):
        if not isinstance(text, str) or not text.strip() or text in ("NaN", "н/д"):
            return np.zeros(nlp.vocab.vectors_length)
        doc = nlp(text)
        vecs = [tok.vector / (np.linalg.norm(tok.vector) + 1e-9)
                for tok in doc if tok.has_vector and tok.is_alpha]
        if not vecs:
            v = doc.vector
            norm = np.linalg.norm(v)
            return v / norm if norm > 0 else np.zeros(nlp.vocab.vectors_length)
        return np.mean(vecs, axis=0)

    def gender_scores(text):
        vec = embed_text(text).reshape(1, -1)
        m = float(cosine_similarity(vec, [male_vec])[0][0])
        f = float(cosine_similarity(vec, [female_vec])[0][0])
        return m, f, m - f

    # 3. Считаем скоры
    print("Computing MCC gender scores...", flush=True)
    rows_mcc = []
    for _, r in mcc_map.iterrows():
        m, f, d = gender_scores(r["mcc_code_desc"])
        rows_mcc.append({"mcc_code": r["mcc_code"], "mcc_code_desc": r["mcc_code_desc"],
                         "mcc_male_sim": m, "mcc_female_sim": f, "mcc_gender_diff": d})
    mcc_scores = pd.DataFrame(rows_mcc)

    print("Computing TR gender scores...", flush=True)
    rows_tr = []
    for _, r in tr_map.iterrows():
        m, f, d = gender_scores(r["tr_type_desc"])
        rows_tr.append({"tr_type": r["tr_type"], "tr_type_desc": r["tr_type_desc"],
                        "tr_male_sim": m, "tr_female_sim": f, "tr_gender_diff": d})
    tr_scores = pd.DataFrame(rows_tr)

    mcc_scores.to_csv("results/gender_embeddings/mcc_gender.csv", index=False, float_format="%.6f")
    tr_scores.to_csv("results/gender_embeddings/tr_gender.csv",  index=False, float_format="%.6f")
    print("Saved → results/gender_embeddings/", flush=True)

    print("\nTop-10 MALE MCC:")
    for _, r in mcc_scores.nlargest(10, "mcc_gender_diff").iterrows():
        print(f"  [{int(r['mcc_code']):3d}] {r['mcc_code_desc'][:55]:<55} diff={r['mcc_gender_diff']:+.3f}")

    print("\nTop-10 FEMALE MCC:")
    for _, r in mcc_scores.nsmallest(10, "mcc_gender_diff").iterrows():
        print(f"  [{int(r['mcc_code']):3d}] {r['mcc_code_desc'][:55]:<55} diff={r['mcc_gender_diff']:+.3f}")

    return mcc_scores, tr_scores


if __name__ == "__main__":
    main()
