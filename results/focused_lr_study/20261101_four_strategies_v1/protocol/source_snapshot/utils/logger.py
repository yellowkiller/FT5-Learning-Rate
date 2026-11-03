import csv
from pathlib import Path


class CsvLogger:
    def __init__(self, path: str | Path, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

    def write(self, row: dict) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(row)

