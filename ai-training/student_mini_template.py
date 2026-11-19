"""
embedding_explorer_mini.py -- the Embedding Map, pocket edition

A smaller, simpler version of the classroom tool: 41 real embeddings from
the model all-MiniLM-L6-v2, baked into this one small file. Standard
library only -- nothing to install.

RUN (Windows):   py embedding_explorer_mini.py
         (Mac):  python3 embedding_explorer_mini.py

THINGS TO TRY:
 1. Click "phishing". The list shows its nearest neighbors, ranked by
    real cosine similarity -- all security words. Then click "pizza"
    and compare the list. Meaning is geometry.
 2. Click "bank" (the red one), then click the two white diamonds.
    Both diamonds contain the word "bank" -- but the sentences landed
    in different neighborhoods. Context bends the arrow.
 3. Click any word, then click a second word: the tool measures the
    angle between them. Click again to start over.
 4. Hover any dot: the panel prints that word's actual numbers.

THE WORDS ON THE MAP (41 things total):
    animals:  cat, dog, shark, eagle, spider, whale
    food:     pizza, apple, taco, cheese, coffee, candy
    tech:     password, phishing, firewall, hacker, laptop, wifi
    school:   homework, teacher, exam, essay, library, backpack
    emotions: happy, sad, angry, nervous, proud, calm
    sports:   soccer, tennis, swimming, baseball, boxing, marathon
    wildcard: bank  (red -- it belongs to two worlds)
    diamonds: 4 whole sentences containing "bank" and "apple"
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

def cosine(a, b):
    """Cosine similarity: 1.0 = same direction, ~0 = unrelated, negative = apart."""
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    return dot / math.sqrt(na * nb)

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

def nearest_item(px, py, maxd=12):
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
    nb_items = others[:8]
    for i, x in enumerate(nb_items):
        nb.insert(tk.END, "%2d. %-16s %+0.3f" % (i + 1, x["label"][:16], x["_sim"]))
        light_label(x, "#9babc8")
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
    cntvar.set("%s <-> %s -- click any word to start over" % (a["label"], b["label"]))
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
        cntvar.set("top 8 of %d -- click another word to measure the angle"
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

def main():
    global canvas, nb, line_ids, selA, selB, nb_items, rebuilding
    global namevar, cntvar, cosvar, statusvar, valvar, root
    root = tk.Tk()
    root.title("Embedding Explorer -- pocket edition (real numbers)")
    root.configure(bg="#0a0e17")

    canvas = tk.Canvas(root, width=W, height=H, bg="#0a0e17", highlightthickness=0)
    canvas.grid(row=0, column=0, padx=10, pady=10)

    panel = tk.Frame(root, bg="#111827")
    panel.grid(row=0, column=1, sticky="nw", padx=(0, 10), pady=10)

    def lab(text, color, size=10, bold=False):
        return tk.Label(panel, text=text, bg="#111827", fg=color, anchor="w",
                        font=("Courier", size, "bold" if bold else "normal"))

    namevar = tk.StringVar(value="-")
    tk.Label(panel, textvariable=namevar, bg="#111827", fg="#eef2ff", anchor="w",
             font=("Courier", 16, "bold")).pack(fill="x", padx=10, pady=(12, 0))
    cntvar = tk.StringVar(value="")
    tk.Label(panel, textvariable=cntvar, bg="#111827", fg="#7183a8", anchor="w",
             font=("Courier", 8), wraplength=300).pack(fill="x", padx=10, pady=(2, 6))

    lab("EMBEDDING VALUES (first 10 of 384)", "#00d4ff", 9, True).pack(fill="x", padx=10)
    valvar = tk.StringVar(value="")
    tk.Label(panel, textvariable=valvar, bg="#0a0e17", fg="#7efff5", anchor="w",
             font=("Courier", 8), wraplength=300, justify="left").pack(fill="x", padx=10, pady=(2, 6))

    nb = tk.Listbox(panel, bg="#0a0e17", fg="#cdd9f0", height=10, width=30,
                    font=("Courier", 10), relief="flat", highlightthickness=0,
                    selectbackground="#1a2234", selectforeground="#00d4ff",
                    activestyle="none")
    nb.pack(fill="x", padx=10, pady=4)
    nb.bind("<<ListboxSelect>>", on_nb_select)

    cosvar = tk.StringVar(value="")
    tk.Label(panel, textvariable=cosvar, bg="#111827", fg="#00d4ff", anchor="w",
             font=("Courier", 12, "bold"), wraplength=300).pack(fill="x", padx=10, pady=(6, 2))

    lab("Hover the map. Click words. Read real numbers.", "#7183a8", 9).pack(fill="x", padx=10, pady=(8, 2))
    statusvar = tk.StringVar(value="")
    tk.Label(panel, textvariable=statusvar, bg="#111827", fg="#9babc8", anchor="w",
             font=("Courier", 9), wraplength=300).pack(fill="x", padx=10, pady=(2, 10))

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
