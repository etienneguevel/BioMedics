import os
from pathlib import Path

import pandas as pd
import typer

from biomedics.normalization.fuzzy import FuzzyNormalizer

os.environ["OMP_NUM_THREADS"] = "16"


def normalize_med_cli(
    drug_dict_path: Path,
    input_dir: Path,
    output_dir: Path,
    label_to_normalize: str,
    with_qualifiers: bool,
    method: str,
    threshold: float,
):
    print(f"Extracting med entities from {input_dir}")
    drug_dict = pd.read_pickle(drug_dict_path)
    normalizer = FuzzyNormalizer(
        input_dir,
        drug_dict,
        label_to_normalize,
        with_qualifiers,
        atc_len=7,
    )
    df = normalizer.normalize(method=method, threshold=threshold)
    if not os.path.exists(output_dir.parent):
        os.makedirs(output_dir.parent)
    path_file = output_dir / f"{input_dir.stem}_{drug_dict_path.stem}.json"
    df.to_json(path_file)
    print(f"Successfully saved the found entities at {path_file}.")


if __name__ == "__main__":
    typer.run(normalize_med_cli)
