"""
embedding_explorer.py -- the Embedding Map, student edition

The same real embeddings that power the classroom web page, as a plain
Python program. Everything runs on the standard library -- nothing to pip
install. Every word inside this file is a real 384-number vector from the
model all-MiniLM-L6-v2.

HOW TO RUN (Windows):
    1. Open Command Prompt in the folder where you saved this file
    2. Type:   py embedding_explorer.py
       (on a Mac:  python3 embedding_explorer.py)

HOW TO USE:
    * Click a word   -> its 15 nearest neighbors appear, ranked by cosine
    * Click another  -> the tool measures the angle between the two
    * Click again    -> fresh selection
    * Type in the FIND box + Enter to jump to a word
    * Hover the map to see what you are pointing at

BONUS (optional):   pip install sentence-transformers
    Unlocks ADD YOUR OWN WORD: the script loads the real model and embeds
    any word you type, live, right onto the map. First run downloads
    about 90 MB -- be patient.
"""

import math
import tkinter as tk

EMBED = __EMBED_DATA__

# ------------------------------------------------------------------ data
ITEMS = []
for _w in EMBED["words"]:
    ITEMS.append({"kind": "w", "label": _w["w"], "cat": _w["c"],
                  "x": _w["x"], "y": _w["y"], "v": _w["v"]})
for _s in EMBED["sents"]:
    ITEMS.append({"kind": "s", "label": _s["label"], "cat": "sentence",
                  "x": _s["x"], "y": _s["y"], "v": _s["v"], "full": _s["full"]})
BY_LABEL = {it["label"]: it for it in ITEMS}
CATCOLORS = EMBED["catcolors"]
PC, MEAN, PCSCALE = EMBED["pcs"], EMBED["mean"], EMBED["pcscale"]

def cosine(a, b):
    """Cosine similarity: 1.0 = same direction, ~0 = unrelated, negative = apart."""
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    return dot / math.sqrt(na * nb)

def project(v):
    """Place a fresh 384-number vector onto the same 2D map (PCA)."""
    c = [vi - mi for vi, mi in zip(v, MEAN)]
    return [sum(ci * pi for ci, pi in zip(c, row)) * PCSCALE for row in PC]

def values_line(it):
    """A peek at the raw numbers: the first 10 of the 384."""
    head = ", ".join("%+.3f" % x for x in it["v"][:10])
    return "[%s, ...]  384 total" % head

# ------------------------------------------------------------------ layout
W, H, PAD = 620, 620, 42
_xs = [it["x"] for it in ITEMS]
_ys = [it["y"] for it in ITEMS]
MINX, MAXX, MINY, MAXY = min(_xs), max(_xs), min(_ys), max(_ys)

def to_screen(x, y):
    sx = PAD + (x - MINX) / (MAXX - MINX) * (W - 2 * PAD)
    sy = H - (PAD + (y - MINY) / (MAXY - MINY) * (H - 2 * PAD))
    return sx, sy

for it in ITEMS:
    it["dx"], it["dy"] = to_screen(it["x"], it["y"])

def nearest_item(px, py, maxd=10):
    best, bd = None, maxd
    for it in ITEMS:
        d = math.hypot(it["dx"] - px, it["dy"] - py)
        if d < bd:
            best, bd = it, d
    return best

# ------------------------------------------------------------------ app
line_ids = []
selA = None
selB = None
nb_items = []
rebuilding = False
MODEL = None

def clear_lines():
    for lid in line_ids:
        canvas.delete(lid)
    line_ids.clear()

def set_highlight(it, on):
    canvas.itemconfig(it["_dot"], outline="#00d4ff" if on else "",
                      width=2 if on else 1)

def dim_labels():
    for x in ITEMS:
        canvas.itemconfig(x["_txt"], state="hidden")

def light_label(it, color):
    canvas.itemconfig(it["_txt"], state="normal", fill=color)

def add_line(a, b, sim, width=1):
    lid = canvas.create_line(a["dx"], a["dy"], b["dx"], b["dy"],
                             fill="#ff4757" if sim < 0 else "#00d4ff", width=width)
    line_ids.append(lid)

def fill_nb(it):
    global rebuilding, nb_items
    rebuilding = True
    others = [x for x in ITEMS if x is not it]
    for x in others:
        x["_sim"] = cosine(it["v"], x["v"])
    others.sort(key=lambda x: -x["_sim"])
    nb.delete(0, tk.END)
    nb_items = others[:15]
    for i, x in enumerate(nb_items):
        nb.insert(tk.END, "%2d. %-20s %+0.3f" % (i + 1, x["label"][:20], x["_sim"]))
        light_label(x, "#9babc8")
    lows = [x for x in others if x["_sim"] < 0][-3:][::-1]
    if lows:
        nb.insert(tk.END, "--- pointing away (negative) ---")
        for x in lows:
            nb.insert(tk.END, " -  %-20s %+0.3f" % (x["label"][:20], x["_sim"]))
    nb.selection_clear(0, tk.END)
    rebuilding = False

def on_nb_select(_e):
    if rebuilding:
        return
    sel = nb.curselection()
    if sel and sel[0] < len(nb_items):
        on_click(nb_items[sel[0]])

def show_compare(a, b):
    sim = cosine(a["v"], b["v"])
    deg = math.degrees(math.acos(max(-1.0, min(1.0, sim))))
    cosvar.set("%+.3f   %.0f degrees apart" % (sim, deg))
    cntvar.set("%s <-> %s  --  pair measured; click any word to start over"
               % (a["label"], b["label"]))
    clear_lines()
    add_line(a, b, sim, 2)

def on_click(it):
    global selA, selB
    if selA is None or selB is not None:
        if selA is not None:
            set_highlight(selA, False)
        if selB is not None:
            set_highlight(selB, False)
        dim_labels()
        selA, selB = it, None
        set_highlight(it, True)
        light_label(it, "#eef2ff")
        clear_lines()
        namevar.set(it["label"])
        cntvar.set("top 15 of %d -- click another word to measure the angle"
                   % (len(ITEMS) - 1))
        cosvar.set("")
        valvar.set(values_line(it))
        fill_nb(it)
    elif it is selA:
        selA = None
        set_highlight(it, False)
        dim_labels()
        namevar.set("-")
        cntvar.set("")
        cosvar.set("")
        valvar.set("")
        nb.delete(0, tk.END)
        clear_lines()
    else:
        selB = it
        set_highlight(it, True)
        dim_labels()
        light_label(selA, "#eef2ff")
        light_label(it, "#eef2ff")
        show_compare(selA, it)
        valvar.set(values_line(it))

def on_canvas_click(e):
    it = nearest_item(e.x, e.y)
    if it is not None:
        on_click(it)

def on_motion(e):
    it = nearest_item(e.x, e.y)
    if it is not None:
        statusvar.set("%s  --  %s" % (it["label"], it.get("full", it["cat"])))
        valvar.set(values_line(it))
    else:
        statusvar.set("")
        valvar.set("")

def on_find(_e=None):
    q = findvar.get().strip().lower()
    if not q:
        return
    hit = None
    for x in ITEMS:
        if q in x["label"].lower():
            hit = x
            break
    if hit is not None:
        on_click(hit)
    else:
        statusvar.set("no word contains '%s' -- look at the map for ideas" % q)

def run_exp(word, si):
    s = EMBED["sents"][si]
    bare, sent = BY_LABEL[word], BY_LABEL[s["label"]]
    on_click(bare)
    on_click(sent)
    sim = cosine(bare["v"], sent["v"])
    deg = math.degrees(math.acos(max(-1.0, min(1.0, sim))))
    scored = [(cosine(sent["v"], x["v"]), x) for x in ITEMS
              if x is not sent and x["kind"] == "w"]
    scored.sort(key=lambda p: -p[0])
    top3 = ", ".join("%s %.2f" % (x["label"], sc) for sc, x in scored[:3])
    expvar.set('bare "%s" vs "%s" -> %.0f degrees. Landed near: %s'
               % (word, s["full"], deg, top3))

def on_add(_e=None):
    global MODEL
    if not HAS_MODEL:
        statusvar.set("optional unlock: run  pip install sentence-transformers  then reopen")
        return
    w = addvar.get().strip()
    if not w:
        return
    if w in BY_LABEL:
        statusvar.set('"%s" is already on the map -- selecting it' % w)
        on_click(BY_LABEL[w])
        return
    statusvar.set("loading the real model (first time: a long download)...")
    root.update()
    try:
        if MODEL is None:
            MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        v = MODEL.encode([w], normalize_embeddings=True)[0].tolist()
    except Exception as ex:
        statusvar.set("model failed: %s" % ex)
        return
    px, py = project(v)
    it = {"kind": "w", "label": w, "cat": "your words", "x": px, "y": py, "v": v}
    it["dx"], it["dy"] = to_screen(px, py)
    it["dx"] = min(max(it["dx"], 8), W - 8)
    it["dy"] = min(max(it["dy"], 8), H - 8)
    ITEMS.append(it)
    BY_LABEL[w] = it
    iid = canvas.create_oval(it["dx"] - 4, it["dy"] - 4, it["dx"] + 4, it["dy"] + 4,
                             fill="#ffffff", outline="")
    tid = canvas.create_text(it["dx"], it["dy"] + 14, text=w, fill="#eef2ff",
                             font=("Courier", 8), state="hidden")
    it["_dot"], it["_txt"] = iid, tid
    statusvar.set('"%s" embedded live and added (white dot)' % w)
    on_click(it)

def main():
    global canvas, nb, line_ids, selA, selB, nb_items, rebuilding
    global namevar, cntvar, cosvar, expvar, statusvar, findvar, addvar, valvar, root
    root = tk.Tk()
    root.title("Embedding Explorer -- every number is real")
    root.configure(bg="#0a0e17")

    canvas = tk.Canvas(root, width=W, height=H, bg="#0a0e17", highlightthickness=0)
    canvas.grid(row=0, column=0, padx=10, pady=10)

    panel = tk.Frame(root, bg="#111827")
    panel.grid(row=0, column=1, sticky="nw", padx=(0, 10), pady=10)

    def lab(parent, text, color, size=10, bold=False):
        return tk.Label(parent, text=text, bg="#111827", fg=color, anchor="w",
                        font=("Courier", size, "bold" if bold else "normal"))

    lab(panel, "FIND A WORD", "#00d4ff", 10, True).pack(fill="x", padx=10, pady=(10, 2))
    findvar = tk.StringVar()
    fe = tk.Entry(panel, textvariable=findvar, bg="#1a2234", fg="#eef2ff",
                  insertbackground="#eef2ff", relief="flat", font=("Courier", 11))
    fe.pack(fill="x", padx=10)
    fe.bind("<Return>", on_find)

    namevar = tk.StringVar(value="-")
    tk.Label(panel, textvariable=namevar, bg="#111827", fg="#eef2ff", anchor="w",
             font=("Courier", 16, "bold")).pack(fill="x", padx=10, pady=(12, 0))
    cntvar = tk.StringVar(value="")
    tk.Label(panel, textvariable=cntvar, bg="#111827", fg="#7183a8", anchor="w",
             font=("Courier", 8), wraplength=310).pack(fill="x", padx=10, pady=(2, 6))

    lab(panel, "EMBEDDING VALUES (first 10 of 384)", "#00d4ff", 9, True).pack(fill="x", padx=10)
    valvar = tk.StringVar(value="")
    tk.Label(panel, textvariable=valvar, bg="#0a0e17", fg="#7efff5", anchor="w",
             font=("Courier", 8), wraplength=310, justify="left").pack(fill="x", padx=10, pady=(2, 6))

    nb = tk.Listbox(panel, bg="#0a0e17", fg="#cdd9f0", height=17, width=34,
                    font=("Courier", 10), relief="flat", highlightthickness=0,
                    selectbackground="#1a2234", selectforeground="#00d4ff",
                    activestyle="none")
    nb.pack(fill="x", padx=10, pady=4)
    nb.bind("<<ListboxSelect>>", on_nb_select)

    cosvar = tk.StringVar(value="")
    tk.Label(panel, textvariable=cosvar, bg="#111827", fg="#00d4ff", anchor="w",
             font=("Courier", 12, "bold"), wraplength=310).pack(fill="x", padx=10, pady=(6, 2))

    lab(panel, "THE CONTEXT EXPERIMENT", "#00d4ff", 10, True).pack(fill="x", padx=10, pady=(10, 2))
    expvar = tk.StringVar(value="pick a sentence chip")
    rows = tk.Frame(panel, bg="#111827")
    rows.pack(fill="x", padx=10)
    for word, idxs in [("bank", (0, 1)), ("apple", (2, 3))]:
        rf = tk.Frame(rows, bg="#111827")
        rf.pack(fill="x", pady=2)
        tk.Label(rf, text=word + " ->", bg="#111827", fg="#eef2ff",
                 font=("Courier", 10, "bold")).pack(side="left")
        for si in idxs:
            s = EMBED["sents"][si]
            tk.Button(rf, text=s["full"], command=lambda si=si, word=word: run_exp(word, si),
                      bg="#1a2234", fg="#cdd9f0", relief="flat", justify="left",
                      font=("Courier", 8), wraplength=190).pack(side="left", padx=3, pady=1, fill="x")
    tk.Label(panel, textvariable=expvar, bg="#111827", fg="#ffd32a", anchor="w",
             font=("Courier", 9), wraplength=310).pack(fill="x", padx=10, pady=4)

    lab(panel, "ADD YOUR OWN WORD (optional)", "#00d4ff", 10, True).pack(fill="x", padx=10, pady=(10, 2))
    addvar = tk.StringVar()
    ae = tk.Entry(panel, textvariable=addvar, bg="#1a2234", fg="#eef2ff",
                  insertbackground="#eef2ff", relief="flat", font=("Courier", 11))
    ae.pack(fill="x", padx=10)
    ae.bind("<Return>", on_add)

    statusvar = tk.StringVar(value="")
    tk.Label(panel, textvariable=statusvar, bg="#111827", fg="#7183a8", anchor="w",
             font=("Courier", 9), wraplength=310).pack(fill="x", padx=10, pady=(4, 10))

    for it in ITEMS:
        X, Y = it["dx"], it["dy"]
        if it["kind"] == "s":
            d = 5
            iid = canvas.create_polygon(X, Y - d, X + d, Y, X, Y + d, X - d, Y,
                                        fill="#ffffff", outline="")
            tid = canvas.create_text(X, Y - 12, text=it["label"], fill="#9babc8",
                                     font=("Courier", 8), state="hidden")
        else:
            iid = canvas.create_oval(X - 4, Y - 4, X + 4, Y + 4,
                                     fill=CATCOLORS.get(it["cat"], "#888"), outline="")
            tid = canvas.create_text(X, Y + 14, text=it["label"], fill="#7183a8",
                                     font=("Courier", 8), state="hidden")
        it["_dot"], it["_txt"] = iid, tid

    canvas.bind("<Button-1>", on_canvas_click)
    canvas.bind("<Motion>", on_motion)
    root.mainloop()

if __name__ == "__main__":
    main()
