"""Download NSL-KDD (with mirrors) into data/nsl_kdd/ if it is missing."""
from pathlib import Path
import urllib.request

DATA = Path(__file__).resolve().parent / "data" / "nsl_kdd"
DATA.mkdir(parents=True, exist_ok=True)

FILES = {
    "KDDTrain+.txt": [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
        "https://raw.githubusercontent.com/4mirhosein/NSL-KDD/master/NSL_KDD_Dataset/KDDTrain%2B.txt",
    ],
    "KDDTest+.txt": [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
        "https://raw.githubusercontent.com/4mirhosein/NSL-KDD/master/NSL_KDD_Dataset/KDDTest%2B.txt",
    ],
}


def download(name, urls):
    target = DATA / name
    if target.exists() and target.stat().st_size > 10000:
        print(f"[OK] {name} already present")
        return
    last = None
    for url in urls:
        try:
            print(f"[DOWNLOAD] {name} <- {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                target.write_bytes(r.read())
            if target.stat().st_size > 10000:
                print(f"[OK] {target.stat().st_size:,} bytes")
                return
        except Exception as exc:  # try the next mirror
            last = exc
            print(f"[WARN] {exc}")
    raise RuntimeError(
        f"Could not download {name}. Place it manually in {DATA} and rerun."
    ) from last


for _name, _urls in FILES.items():
    download(_name, _urls)
