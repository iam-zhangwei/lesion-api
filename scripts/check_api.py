import pathlib

import httpx

for f in sorted(pathlib.Path("check_images").glob("*.jpg"),
                key=lambda p: int(p.name.split("_")[0])):
    with open(f, "rb") as fh:
        r = httpx.post("http://localhost:8000/predict",
                       files={"file": (f.name, fh, "image/jpeg")}, timeout=30)
    d = r.json()
    print(f"{f.name:45s} -> {d['label']:6s} {d['confidence']:.3f}")