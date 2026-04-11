import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r"d:\retina\retina_(1)_(1).ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, c in enumerate(nb["cells"]):
    if i < 12:
        continue
    src = "".join(c["source"])[:800]
    print(f"=== CELL {i} ({c['cell_type']}) ===")
    print(src)
    print()
