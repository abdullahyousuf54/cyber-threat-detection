"""Restore large, bundled demo assets when a cloud host starts."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile


ASSET_PATHS = (
    "data/nsl_kdd/KDDTrain+.txt",
    "data/nsl_kdd/KDDTest+.txt",
    "models/nslkdd/random_forest.joblib",
    "models/nslkdd/isolation_forest.joblib",
    "models/nslkdd/evaluation.json",
    "results/evaluation.json",
)

ASSET_SIZES = (19109424, 3441513, 24183089, 1506441, 2110188, 2110188)
ARCHIVE_SHA256 = "5f10fa5e971a7606a4695174951ec6e2b57b53068f7cadcd56d14a2f62b0b9d2"


def ensure_assets(root):
    root = Path(root)
    if all(
        (root / name).is_file() and (root / name).stat().st_size == size
        for name, size in zip(ASSET_PATHS, ASSET_SIZES)
    ):
        return

    parts = sorted((root / "assets").glob("part-*.bin"))
    if not parts:
        raise FileNotFoundError("Bundled model and dataset assets are missing")

    archive = b"".join(part.read_bytes() for part in parts)
    if sha256(archive).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Bundled model and dataset assets failed verification")

    with ZipFile(BytesIO(archive)) as bundle:
        if set(bundle.namelist()) != set(ASSET_PATHS):
            raise ValueError("Bundled asset list does not match the manifest")
        for name in ASSET_PATHS:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(name) as source, target.open("wb") as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
