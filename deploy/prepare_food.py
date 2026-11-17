"""Install the four FNDDS tables used by IACES from the USDA's dated public release."""
import hashlib
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

URL = "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_survey_food_csv_2024-10-31.zip"
TABLES = {
    "food.csv": "3d415f2a2c44e2dbb799a7b6a2a2b5fc05dce835dafee72fd54319b0ef6d9a28",
    "food_portion.csv": "994a9469a0319a378a887bf200be44cde33166303c8cbf6e5debdc8814c90006",
    "food_nutrient.csv": "1703575ccff450166efe12cc06bf1245364e9fa11a6175c120fb9233f922d006",
    "nutrient.csv": "dbdf1b819de4a87abdbf0275402860881d35dfa6e4bc418c34d203e5e2d5b923",
}
TARGET = Path(__file__).resolve().parents[1] / "data/raw/fndds-2024-10"


def valid(folder):
    for name, expected in TABLES.items():
        path = folder / name
        if not path.is_file():
            return False
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != expected:
                return False
    return True


def main():
    if valid(TARGET):
        print("FNDDS nutrition tables are ready.")
        return
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TARGET.parent) as tmp:
        stage = Path(tmp)
        with urllib.request.urlopen(URL, timeout=120) as response, (stage / "fdc.zip").open("wb") as out:
            shutil.copyfileobj(response, out)
        with zipfile.ZipFile(stage / "fdc.zip") as archive:
            for info in archive.infolist():
                name = Path(info.filename).name
                if name in TABLES:
                    with archive.open(info) as source, (stage / name).open("wb") as dest:
                        shutil.copyfileobj(source, dest)
        if not valid(stage):
            raise ValueError("The USDA download does not match the four expected October 2024 FNDDS tables.")
        TARGET.mkdir(exist_ok=True)
        for name in TABLES:
            (stage / name).replace(TARGET / name)
    print("Installed and verified four USDA FNDDS nutrition tables.")


if __name__ == "__main__":
    main()
