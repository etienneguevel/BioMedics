import os
from typing import List, Optional, Tuple

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
        [[k, v] for k, v in ai_bio_dict.items()], columns=["norm_term", "normalized_label"]  # noqa: E501
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


def main(config_path: str, output_path: Optional[str] = None):
    # Read the config file
    try:
        config = OmegaConf.load(config_path)
    except Exception as e:
        print(f"Error loading config file: {e}")
        raise typer.Exit(code=1)

    # Check the output_path
    output_path = output_path or config.data.output
    if output_path is None:
        raise ValueError("The output path is required.")

    elif output_path.endswith(".parquet"):
        pass

    else:
        output_path += ".parquet"

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
    # df_other = df_ents[~(df_ents.label.isin(["BIO_comp", "Chemical_and_drugs"]))]

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

    df_bio = df_bio.explode(["norm_term"])
    df_drug = df_drug.explode(["norm_term"])

    # Clean the data
    df_bio = clean_bio_df(df_bio)
    df_drug = clean_drug_df(df_drug)

    # Concatenate the different entities
    df_ents = pd.concat([
        df_bio,
        df_drug,
    ], axis=0)
    df_ents = df_ents.rename(columns={"value_cleaned": "value"})

    # Save the processed data
    print(f"Saving at {output_path}")
    df_ents.to_parquet(output_path)


if __name__ == "__main__":
    typer.run(main)
