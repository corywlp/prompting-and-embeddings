"""
embed_lesson_data.py — generates the data baked into prompting-and-embeddings.html

What it does, in order:
  1. Embeds ~92 words (grouped by category) + 4 full sentences with the same
     model the HTML teaches about: all-MiniLM-L6-v2 (384-dim vectors).
  2. Runs PCA (via SVD) to flatten the 384-dim arrows down to 2D for the map.
     The full 384-dim vectors ALSO go into the HTML — the neighbor lists and
     cosine scores there are computed on the real vectors, not the projection.
  3. Injects everything into the HTML template as one JSON object.

Re-run any time you edit the word lists:
    source ai-lab/bin/activate   (if the terminal is new)
    python ai-training/embed_lesson_data.py

The dictionary below is the whole curriculum — edit it and the tool rebuilds.
"""

import json
import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer

HTML_PATH = "/Users/coryzoblin/Desktop/claude/prompting-and-embeddings.html"

# --------------------------------------------------------------------------
# 1. The vocabulary — each category gets its own color on the map.
#    "wildcards" are words with double meanings: which world wins?
# --------------------------------------------------------------------------
VOCAB = {
    "animals": ["cat", "dog", "lion", "tiger", "shark", "eagle", "salmon", "whale", "spider", "horse", "turtle", "crab"],
    "food": ["pizza", "sushi", "burger", "salad", "taco", "apple", "banana", "cheese", "coffee", "candy"],
    "sports": ["soccer", "basketball", "tennis", "boxing", "swimming", "marathon", "baseball", "hockey"],
    "emotions": ["happy", "sad", "angry", "excited", "nervous", "proud", "jealous", "calm", "terrified", "joyful"],
    "tech & security": ["password", "firewall", "phishing", "malware", "encryption", "hacker", "ransomware", "laptop", "server", "wifi"],
    "school": ["homework", "teacher", "exam", "essay", "classroom", "diploma", "student", "library", "notebook", "detention", "backpack"],
    "vehicles": ["car", "truck", "airplane", "motorcycle", "submarine", "helicopter", "bicycle", "spaceship"],
    "money": ["money", "loan", "deposit", "salary", "invoice", "budget", "cash", "vault", "teller", "credit"],
    "nature": ["river", "lake", "ocean", "mountain", "forest", "creek", "waterfall", "shore", "canoe", "fishing", "seashell", "beach"],
    "wildcards": ["python", "jaguar", "shell", "mouse", "bat", "port", "bank"],
    "body": ["hand", "foot", "eye", "heart", "brain", "tooth", "hair", "knee", "blood", "bone"],
    "music": ["guitar", "piano", "drum", "violin", "song", "concert", "chorus", "rhythm", "melody", "trumpet"],
    "weather": ["rain", "snow", "thunder", "hurricane", "fog", "breeze", "drought", "lightning", "cloud", "hail"],
    "jobs": ["doctor", "nurse", "firefighter", "plumber", "chef", "pilot", "lawyer", "farmer", "mechanic", "janitor"],
    "plants": ["tree", "flower", "grass", "rose", "cactus", "oak", "leaf", "seed", "garden", "weed"],
    "clothing": ["shirt", "jeans", "sneakers", "jacket", "hat", "dress", "socks", "scarf", "helmet", "gloves"],
    "tools": ["hammer", "screwdriver", "wrench", "drill", "saw", "ladder", "tape", "glue", "rope", "bolt"],
    "space": ["planet", "star", "moon", "comet", "galaxy", "astronaut", "rocket", "telescope", "gravity", "orbit"],
    "family": ["mother", "father", "sister", "brother", "grandmother", "cousin", "baby", "neighbor", "friend", "wedding"],
    "games": ["chess", "checkers", "puzzle", "dice", "cards", "videogame", "riddle", "toy", "lottery", "maze"],
    "kitchen": ["oven", "fork", "knife", "plate", "fridge", "microwave", "toaster", "pan", "spoon", "blender"],
}

CATCOLORS = {
    "animals": "#00ff88",
    "food": "#ffd32a",
    "sports": "#ff6b35",
    "emotions": "#ff6ba9",
    "tech & security": "#00d4ff",
    "school": "#8899ff",
    "vehicles": "#aaff00",
    "money": "#00b8a9",
    "nature": "#7c3aed",
    "wildcards": "#ff4757",
    "body": "#f8a5c2",
    "music": "#c56cf0",
    "weather": "#7efff5",
    "jobs": "#a4b0be",
    "plants": "#6ab04c",
    "clothing": "#fab1a0",
    "tools": "#e77f67",
    "space": "#5352ed",
    "family": "#f7d794",
    "games": "#70a1ff",
    "kitchen": "#eccc68",
}

# Same keyword, two different sentences — watch the arrows move (tab 3).
SENTENCES = [
    ("bank (river)", "We had a picnic by the river bank."),
    ("bank (money)", "I deposited my paycheck at the bank."),
    ("apple (fruit)", "I ate a crisp apple at lunch."),
    ("apple (tech)", "Apple announced a brand new iPhone."),
]

# --------------------------------------------------------------------------
# 2. Embed everything with the real model.
# --------------------------------------------------------------------------
def main():
    words = [(w, cat) for cat, ws in VOCAB.items() for w in ws]
    texts = [w for w, _ in words] + [s for _, s in SENTENCES]
    labels = [f"{w} [{cat}]" for w, cat in words] + [f"SENT: {s}" for s, _ in SENTENCES]

    print(f"Loading all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Embedding {len(texts)} texts (384 dims each) ...")
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=float)
    print(f"Got matrix: {emb.shape}")

    # PCA: flatten 384 dims -> 2D, fitted on ALL vectors together so words and
    # sentences share one map. SVD does the work; Vt rows are the components.
    # pcs/mean/pcscale ship with the data so NEW words can be projected live
    # (the student script's "add your own word" feature).
    centered = emb - emb.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    raw = centered @ vt[:2].T
    pcscale = 100.0 / float(np.abs(raw).max())
    proj = raw * pcscale

    data = {
        "catcolors": CATCOLORS,
        "pcs": [[round(float(x), 4) for x in row] for row in vt[:2]],
        "mean": [round(float(x), 4) for x in emb.mean(axis=0)],
        "pcscale": round(pcscale, 6),
        "words": [
            {"w": w, "c": cat, "x": round(float(proj[i, 0]), 2), "y": round(float(proj[i, 1]), 2),
             "v": [round(float(x), 3) for x in emb[i]]}
            for i, (w, cat) in enumerate(words)
        ],
        "sents": [
            {"label": label, "full": full, "x": round(float(proj[len(words) + i, 0]), 2),
             "y": round(float(proj[len(words) + i, 1]), 2),
             "v": [round(float(x), 3) for x in emb[len(words) + i]]}
            for i, (label, full) in enumerate(SENTENCES)
        ],
    }

    # --------------------------------------------------------------------------
    # 3. Sanity checks — printed so you can see the model agreeing with reality.
    # --------------------------------------------------------------------------
    def neighbors(label, n=4):
        i = labels.index(label)
        sims = emb @ emb[i]  # normalized vectors: dot product IS cosine
        order = np.argsort(-sims)
        return [(labels[j].split(" [")[0], round(float(sims[j]), 3)) for j in order[1:n + 1]]

    print("\nSanity checks (real cosine, full 384 dims):")
    for probe in ["phishing [tech & security]", "python [wildcards]", "shell [wildcards]"]:
        print(f"  {probe.split(' [')[0]} → {neighbors(probe)}")
    i_river = labels.index("SENT: bank (river)"); i_money = labels.index("SENT: bank (money)")
    j_river = [w for w, _ in words].index("river"); j_loan = [w for w, _ in words].index("loan")
    print(f"  cos('bank (river)' sentence, 'river') = {emb[i_river] @ emb[j_river]:.3f}")
    print(f"  cos('bank (money)' sentence, 'loan')  = {emb[i_money] @ emb[j_loan]:.3f}")

    # --------------------------------------------------------------------------
    # 4. Inject into the HTML template (re-runnable: replaces old data too),
    #    then build the standalone student script from the same data.
    # --------------------------------------------------------------------------
    payload = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    with open(HTML_PATH) as f:
        html = f.read()
    new_html, n = re.subn(r"const EMBED = .*?;\n", f"const EMBED = {payload};\n", html, flags=re.S)
    if n != 1:
        raise SystemExit(f"Expected exactly one EMBED line in the template, found {n} — aborting, file untouched.")

    # 4b. Student script: same data, tkinter, stdlib-only, Windows-friendly.
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "student_explorer_template.py")) as f:
        tpl = f.read()
    py_src, n2 = re.subn(r"^EMBED = __EMBED_DATA__$", "EMBED = " + payload, tpl, count=1, flags=re.M)
    if n2 != 1:
        raise SystemExit("student template placeholder missing — aborting, nothing written.")
    out_py = "/Users/coryzoblin/Desktop/claude/embedding_explorer.py"
    with open(out_py, "w") as f:
        f.write(py_src)

    # 4c. Drop the SAME template (placeholder intact) into the HTML tab 4,
    #     so the download button can bake the data in the student's browser.
    marker = "the Embedding Map, student edition"
    if "<!--STUDENT_CODE-->" in new_html:
        new_html, n3 = re.subn("<!--STUDENT_CODE-->", tpl, new_html, count=1)
        if n3 != 1:
            raise SystemExit("student-code anchor broken — aborting.")
    elif marker not in new_html:
        raise SystemExit("no <!--STUDENT_CODE--> anchor in HTML — aborting.")
    print(f"\nInjected {len(payload) // 1024} KB of real embeddings into {HTML_PATH}")
    print(f"Wrote student script: {out_py} ({len(py_src) // 1024} KB)")

    # 5. Pocket edition: same vectors, same map coordinates, 41 items.
    #    Built as a strict SUBSET of the full data (in full-data word order),
    #    so the browser download can reproduce it byte-for-byte by filtering
    #    EMBED — see MINI_WORDS / miniData() in the HTML. Keep the two
    #    word lists in sync.
    MINI_VOCAB = {
        "animals": ["cat", "dog", "shark", "eagle", "spider", "whale"],
        "food": ["pizza", "apple", "taco", "cheese", "coffee", "candy"],
        "tech & security": ["password", "phishing", "firewall", "hacker", "laptop", "wifi"],
        "school": ["homework", "teacher", "exam", "essay", "library", "backpack"],
        "emotions": ["happy", "sad", "angry", "nervous", "proud", "calm"],
        "sports": ["soccer", "tennis", "swimming", "baseball", "boxing", "marathon"],
        "wildcards": ["bank"],
    }
    mini_labels = {w for ws in MINI_VOCAB.values() for w in ws}
    mdata = {
        "catcolors": {k: CATCOLORS[k] for k in MINI_VOCAB},
        "words": [w for w in data["words"] if w["w"] in mini_labels],
        "sents": data["sents"][:4],
    }
    assert len(mdata["words"]) == 37, f"mini subset has {len(mdata['words'])} words, expected 37"
    mpayload = json.dumps(mdata, ensure_ascii=True, separators=(",", ":"))
    with open(os.path.join(here, "student_mini_template.py")) as f:
        mtpl = f.read()
    msrc, n4 = re.subn(r"^EMBED = __EMBED_DATA__$", "EMBED = " + mpayload, mtpl, count=1, flags=re.M)
    if n4 != 1:
        raise SystemExit("mini template placeholder missing — aborting, nothing written.")
    out_mini = "/Users/coryzoblin/Desktop/claude/embedding_explorer_mini.py"
    with open(out_mini, "w") as f:
        f.write(msrc)
    if "<!--STUDENT_MINI_CODE-->" in new_html:
        new_html, n5 = re.subn("<!--STUDENT_MINI_CODE-->", mtpl, new_html, count=1)
        if n5 != 1:
            raise SystemExit("mini anchor broken — aborting.")
    elif "pocket edition" not in new_html:
        raise SystemExit("no <!--STUDENT_MINI_CODE--> anchor in HTML — aborting.")
    with open(HTML_PATH, "w") as f:
        f.write(new_html)
    print(f"Wrote mini student script: {out_mini} ({len(msrc) // 1024} KB)")

if __name__ == "__main__":
    main()
