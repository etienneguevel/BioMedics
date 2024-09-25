import os
from typing import List, Tuple

import pandas as pd
import typer
from omegaconf import OmegaConf

from biomedics.extract_measurement.main import main as extract_measurements
from biomedics.ner.extract import main as extract_ents
from biomedics.normalization.coder_inference.main import main as coder_inference
from biomedics.normalization.fuzzy.main import main as fuzzy_normalize


def clean_bio_df(df: pd.DataFrame) -> pd.DataFrame:

    # Create a dictionary to select only the relevant terms
    ai_bio_dict = {
        "ca19.9": "C0201551",
        "platelets": "C0032181",
        "creatinine": "C0201975",
        "bilirubin": ["C0201913", "C0201916", "C0201914", "C0523531"],
        "hemoglobin": "C0518015",
        "albumin": ["C0042038", "C0201838"],
        "ACE": "C0201888",
        "AST": "C0201899",
        "ALT": "C0201836",
        "LDH": "C0202113",
        "CRP": "C0201657",
        "dpd": "C0523601",
        "granulocyte_count": "C0857490",
        "lymphocyte_count": "C0200635",
        "leukocyte_count": "C0023508",
        "neutrophile_count": "C0200633",
    }
    df_bio_dict = pd.DataFrame(
        [[k, v] for k, v in ai_bio_dict.items()], columns=["norm_term", "normalized_label"]
    ).explode(["normalized_label"])

    df = df.drop(columns=["norm_term"])
    df = df.merge(df_bio_dict, on="normalized_label", how="inner")
    return df

def clean_drug_df(df: pd.DataFrame) -> pd.DataFrame:

    cure_terms = ["folfirinox", "folfox", "folfiri", "xelox", "capox"]
    df = df[
        (df.normalized_label.str.startswith("L"))
        | (df.normalized_label.isin(cure_terms))
    ]
    return df

def load_texts(root: str) -> List[Tuple[str, str]]:
    if os.path.isdir(root):
        texts_path = [
            os.path.join(root, f) for f in os.listdir(root) if f.endswith(".txt")
        ]
        texts = [
            (
                f.split("/")[-1].replace(".txt", ""),
                open(f, "r").read()
            )
            for f in texts_path
        ]

    elif root.endswith(".csv"):
        df = pd.read_csv(root)
        texts = [(str(row["note_id"]), str(row["note_txt"])) for _, row in df.iterrows()]

    else:
        raise ValueError("The root must be a directory or a file.")

    return texts

def make_ann_file(df_ents, note_id, attr=None):
    if attr is None:
        attr = []

    def replace_newline(text):
        return text.replace('\n', ' ')

    s = ""
    a = 1
    e = 1

    for i, ent in df_ents[df_ents.source == note_id].iterrows():
        s += f"T{e}\t{ent['label']} {ent['span_start']} {ent['span_end']}\t{replace_newline(ent['norm_term'])}\n"

        for k in attr:
            if ent[k] is not None:
                s += f"A{a}\t{k} T{e} {ent.get(k)}\n"
                a += 1
        e += 1

    return s

def main(config_path: str):
    # Read the config file
    config = OmegaConf.load(config_path)

    # Extract entities from the texts
    extract_ents(config.data.root, config.data.raw_output)

    # Extract measurements
    df_ents = extract_measurements(
        config.measurements,
        config.data.raw_output,
        attributes=config.attributes,
    )
    df_bio = df_ents[df_ents.label == "BIO_comp"]
    df_drug = df_ents[df_ents.label == "Chemical_and_drugs"]
    df_other = df_ents[~(df_ents.label.isin(["BIO_comp", "Chemical_and_drugs"]))]

    del df_ents
    # Normalize the BIO measurements
    df_bio = coder_inference(
        df_bio,
        config.normalization.BIO,
    )
    print(df_bio.columns)

    #Normalize the chemical_and_drugs measurements
    df_drug = fuzzy_normalize(
        df_drug,
        config.normalization.chemical_and_drugs,
    )

    # Concatenate the different entities
    df_ents = pd.concat([
        df_bio,
        df_drug,
        df_other
    ], axis=0)

    df_ents = df_ents.explode(["norm_term"])

    # Save the processed data
    if config.data.format == "parquet":
        df_ents.to_parquet(config.data.output)

    elif config.data.format == "brat":
        if not os.path.isdir(config.data.output):
            os.makedirs(config.data.output)

        # Clean the data
        df_bio = clean_bio_df(df_bio)
        df_drug = clean_drug_df(df_drug)
        df_ents = pd.concat([df_bio, df_drug], axis=0)

        # Load the texts
        texts = load_texts(config.data.root)
        for note_id, text in texts:
            ann_file = make_ann_file(df_ents, note_id, attr=config.attributes)
            with open(os.path.join(config.data.output, f"{note_id}.ann"), "w") as f:
                f.write(ann_file)
            with open(os.path.join(config.data.output, f"{note_id}.txt"), "w") as f:
                f.write(text)

        pass
    else:
        raise ValueError("The format indicated is not recognized.")

if __name__ == "__main__":
    typer.run(main)
