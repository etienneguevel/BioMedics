import typer

import pandas as pd
from omegaconf import OmegaConf

from biomedics.extract_measurement.main import main as extract_measurements
from biomedics.ner.extract import main as extract_ents
from biomedics.normalization.coder_inference.main import main as coder_inference
from biomedics.normalization.fuzzy.main import main as fuzzy_normalize


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
        # ToDo: write a function to save to brat format
        
        
        pass
    else:
        raise ValueError("The format indicated is not recognized.")

if __name__ == "__main__":
    typer.run(main)
