import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from biomedics.extract_measurement.cleaner import (
    _convert_brat_spans,
    clean_lexical_variant,
    extract_clean_non_digit_value,
    extract_clean_range_value,
    extract_clean_subsequent_lex_var,
    extract_clean_units,
    extract_clean_values,
    match_bio_to_biocomp,
    match_date_pattern,
)
from biomedics.utils.extract_pandas_from_brat import extract_pandas


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from a path and convert to a pandas dataframe.
    Accepts .ann files or .csv files or .parquet files.
    Args:
        data_path (str): Path to the data.
        labels (List[str]): List of labels to filter.
    Returns:
        pd.DataFrame: Dataframe with the data.
    """
    if os.path.isdir(data_path):
        df = extract_pandas(IN_BRAT_DIR=data_path)
        df[['span_start', 'span_end']] = df['span'].apply(_convert_brat_spans).tolist()
        df["lexical_variant"] = df["term"].copy()

    elif data_path.endswith(".csv"):
        df = pd.read_csv(data_path)

    elif data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
        df = df.rename(columns={
            'start': 'span_start',
            'end': 'span_end',
            'note_id': 'source',
        })
        df["term"] = df["lexical_variant"].copy()

    else:
        raise ValueError(f"Invalid data path: {data_path}")


    return df


def main(
    script_config: Dict[str, Any],
    data_path: str,
    output_dir: Optional[str]=None,
    attributes: Optional[List[str]]=None
) -> pd.DataFrame:
    """
    Pipeline to apply measurement extraction to entities extracted
    with the EDS-Biomedic model.
    Args:
        - script_config (Dict[str, Any]) : configuration setup to use
        - data_path (str) : path to the ents, should be a .csv, .parquet or a dir_path
        - output_dir (Optional[str]) : path to the directory where the extracted ents
        will be saved
        - attributes (Optional[List[str]]) : attributes to add to the extracted entities

    Returns:
    A dataframe with the BIO ents and their extracted measurements.
    """
    if attributes is None:
        attributes = []

    logger.info('-------------Load entities-------------')

    start_t1 = time.time()

    label_key = script_config["label_key"]
    labels_to_remove = script_config["labels_to_remove"]
    all_labels = [label_key] + list(labels_to_remove)

    # Load dataset
    df = load_data(data_path)

    # Retrieve the BIO and BIO_comp entities within the dataset
    df_ents = df.copy()
    df_ents = df_ents[df_ents["label"].isin(all_labels)].drop_duplicates()

    df_ents_bio_comp = df_ents[df_ents["label"] == label_key]
    df_ents_bio = df_ents[df_ents["label"].isin(labels_to_remove)]
    end_t1 = time.time()

    logger.info(
        f"Biocomp table len:  {len(df_ents_bio_comp)} entities\
     {df_ents_bio_comp.source.nunique()} docs\
     \nBio table len: {len(df_ents_bio)}\
     {df_ents_bio.source.nunique()} docs\
     \nProcessed in {round(end_t1 - start_t1,3)} secs"
    )

    logger.info('-------------Link bio to bio_comp-------------')
    start_t2 = time.time()
    df_biocomp_bio = match_bio_to_biocomp(df_ents_bio, df_ents_bio_comp)
    end_t2 = time.time()

    logger.info(
        f"""Biocomp linked with bio table len:  {len(df_biocomp_bio)} entities
        processed in {round(end_t2 - start_t2,3)} secs"""
    )

    logger.info('-------------Remove bio from bio_comp-------------')
    start_t3 = time.time()

    # Remove values in 'lexical_variant_biocomp' that are in 'lexical_variant_bio'
    df_biocomp_bio["lexical_variant_stripped"] = df_biocomp_bio.apply(
        lambda row: row["lexical_variant_biocomp"].replace(row["lexical_variant_bio"], "").strip(),
        axis=1
    )

    df_biocomp_bio["lexical_variant_stripped"] = df_biocomp_bio["lexical_variant_stripped"].str.replace(
        r"G\/+[lL]\b", "x10*9/l", regex=True
    )

    end_t3 = time.time()
    logger.info(f"processed in {round(end_t3 - start_t3,3)} secs ---")

    logger.info('-------------Extract the date from ents containing one -------------')
    start_t4 = time.time()
    df_biocomp_bio = match_date_pattern(df_biocomp_bio)
    end_t4 = time.time()
    logger.info(f"processed in {round(end_t4 - start_t4,3)} secs ---")

    logger.info('-------------Clean and extract range value -------------')
    start_t5 = time.time()
    df_biocomp_bio = extract_clean_range_value(df_biocomp_bio)
    end_t5 = time.time()
    logger.info(f"processed in {round(end_t5 - start_t5,3)} secs")

    logger.info('-------------Clean the entities col with pollution-------------')
    start_t6 = time.time()
    df_biocomp_bio_clean = clean_lexical_variant(df_biocomp_bio)
    end_t6 = time.time()
    logger.info(f"processed in {round(end_t6 - start_t6,3)} secs")

    logger.info('-------------Clean and extract non digit values-------------')
    start_t7 = time.time()
    df_biocomp_bio_clean = extract_clean_non_digit_value(df_biocomp_bio_clean)
    end_t7 = time.time()
    logger.info(f"processed in {round(end_t7 - start_t7,3)} secs")

    logger.info('-------------Clean and extract units-------------')
    start_t8 = time.time()
    df_biocomp_bio_clean = extract_clean_units(df_biocomp_bio_clean)
    end_t8 = time.time()
    logger.info(f"processed in {round(end_t8 - start_t8,3)} secs")

    logger.info('-------------Clean and extract values-------------')
    start_t9 = time.time()
    df_biocomp_bio_clean = extract_clean_values(df_biocomp_bio_clean)
    end_t9 = time.time()
    logger.info(f"processed in {round(end_t9 - start_t9,3)} secs")

    logger.info('-------------Clean, extract subsequent lexical variant-------------')
    start_t11 = time.time()
    df_biocomp_bio_clean = extract_clean_subsequent_lex_var(df_biocomp_bio_clean)
    end_t11 = time.time()
    logger.info(f"processed in {round(end_t11 - start_t11,3)} secs")

    logger.info(
        f"Dataframe shape after processing: {df_biocomp_bio_clean.count()}\n"
        f"Number of unique note_id in the initial df: "
        f"{df_biocomp_bio.source.count()}.\n"
        f"After processing: {df_biocomp_bio_clean.source.nunique()}"
    )


    logger.info('---------------Select columns of interest and convert to Pandas---------------')  # noqa: E501
    attributes_found = [a for a in attributes if a in df_biocomp_bio_clean.columns]
    df_final = df_biocomp_bio_clean[
        set([
            'source',
            'span_start',
            'span_start_bio',
            'span_end',
            'span_end_bio',
            'term_biocomp',
            'term_bio',
            'lexical_variant_stripped',
            'range_value',
            # 'non_digit_value',
            'unit',
            'lexical_variant',
            'value_cleaned',
            'extracted_date',
            # 'fluid_source',
        ] + attributes_found)
    ]

    if output_dir:
        logger.info('---------------Save in pickle---------------')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df_final.to_pickle(f"{output_dir}/pred_with_measurement.pkl")

        logger.info('---------------saved---------------')
        logger.info(f"saved path: {output_dir}/pred_with_measurement.pkl")

    final_time = time.time()
    logger.info(f"Total processing time:  {round(final_time - start_t1,3)} secs")

    # Merge with the original dataframe, keep the lexical variant filtered.
    df_final = df_final[
        ["source", "span_start", "span_end", "lexical_variant", "value_cleaned", "range_value", "unit"]
    ].merge(
        df[[c for c in df.columns if c != "lexical_variant"]],
        on=("source", "span_start", "span_end")
    )

    df_out = pd.concat([
        df_final,
        df[~(df["label"].isin(all_labels))]
    ], axis=0)
    df_out["value_cleaned"] = df_out["value_cleaned"].astype(str)

    return df_out
