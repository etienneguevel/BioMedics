import typer
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
    df_ents = extract_measurements(config.measurements, config.data.raw_output)

    # Normalize the BIOmeasurements
    df_ents = coder_inference(
        df_ents,
        config.normalization.BIO,
    )
    print(df_ents.columns)

    #Normalize the chemical_and_drugs measurements
    df_ents = fuzzy_normalize(
        df_ents,
        config.normalization.chemical_and_drugs,
    )

    # Save the processed data
    df_ents.to_parquet(config.data.output)


if __name__ == "__main__":
    typer.run(main)
