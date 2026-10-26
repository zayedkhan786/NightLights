"""Retain a real-data six-day incremental demonstration and machine-readable report."""
import hashlib
import json
from pathlib import Path
import uuid
from update_viirs import update, verify_history
from viirs_zarr import BANDS


def fingerprint(store):
    digest = hashlib.sha256()
    for file in sorted(Path(store).rglob("*")):
        if file.is_file():
            digest.update(str(file.relative_to(store)).encode())
            digest.update(file.read_bytes())
    return digest.hexdigest()


def main():
    root = Path("data/zarr_validation") / ("six_day_" + uuid.uuid4().hex[:8])
    reports = []
    for product in BANDS:
        first = update(product, "2016-01-01", "2016-01-03", output=root)
        extended = update(product, "2016-01-01", "2016-01-06", output=root)
        store = root / f"{product}.zarr"
        before = fingerprint(store)
        repeated = update(product, "2016-01-01", "2016-01-06", output=root)
        after = fingerprint(store)
        assert first["pending_at_start"] == extended["pending_at_start"] == 3
        assert repeated["pending_at_start"] == repeated["local_files_read"] == repeated["nasa_batches"] == 0
        assert before == after
        assert verify_history(product, output=root) == 6
        reports.append({"product": product, "initial": first, "extension": extended,
                        "repeat": repeated, "repeat_store_unchanged": before == after,
                        "verified_days": 6, "store_sha256": after})
    (root / "validation_report.json").write_text(json.dumps(reports, indent=2))
    print(f"Validated six days for both products: {root.resolve()}")


if __name__ == "__main__":
    main()
