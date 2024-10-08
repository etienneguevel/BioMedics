import os
import sys
from typing import Any, List, Optional, Union

import edsnlp
import pandas as pd
import pyarrow.parquet as pq
import spacy
import torch
import typer
from spacy.tokens import Doc


def convert_doc_to_dict(doc: Doc, attributes: Optional[List[str]] = None) -> List[dict]:
    if attributes is None:
        attributes = []
    ents = [
        {**{
            "note_id": doc._.note_id,
            "lexical_variant": e.text,
            "label": e.label_,
            "start": e.start_char,
            "end": e.end_char,
        }, **{
            attr: getattr(e._, attr) for attr in attributes
        }}
        for e in doc.ents
    ]
    if "BIO" in doc.spans:
        spans = [
            {**{
                "note_id": doc._.note_id,
                "lexical_variant": s.text,
                "label": "BIO",
                "start": s.start_char,
                "end": s.end_char,
            }, **{
                attr: getattr(s._, attr) for attr in attributes
            }}
            for s in doc.spans["BIO"]
        ]
    else:
        spans = []
    return ents + spans

def build_data(corpus: Union[str, pd.DataFrame], filter: Optional[Any] = None):
    """
    This function builds a data iterator from a text corpus.
    The data iterator can then be used to map a nlp model to the txts or for other
    functions.
    Args:
        - corpus: either a directory with txts inside, a path to a .csv file that is in
    the form ["note_id", "note_txt"] or a pandas DataFrame with the same columns.
        - filter: function taking a dataframe as en entry, supposed to filter the corpus
        according to defined criterias.
    Returns:
        An iterator of spacy docs of the corpus.
    """
    if isinstance(corpus, str):
        if os.path.isdir(corpus):
            print(f"Building from dir {corpus}")
            data = edsnlp.data.read_standoff(corpus)  # type: ignore

        elif corpus.endswith((".csv", ".parquet")):
            print(f"Loading from {corpus}.")
            if corpus.endswith(".csv"):
                df = pd.read_csv(corpus)
            else:
                df = pq.read_table(corpus).to_pandas(
                    timestamp_as_object=True
                )  # Pandas can fail to read parquet files sometimes
            if filter:
                df = filter(df)

            data = edsnlp.data.from_pandas(df, converter="omop")  # type: ignore

        else:
            raise ValueError("The corpus must be a directory or a readable file.")

    elif isinstance(corpus, pd.DataFrame):
        print("Using corpus as a pandas DataFrame")
        data = edsnlp.data.from_pandas(corpus, converter="omop")  # type: ignore

    else:
        raise TypeError(f"Expected str of pd.DataFrame types, got {type(corpus)}")

    return data


def extract_ents_from_docs(
    docs,
    nlp: edsnlp.Pipeline,
    attributes: Optional[List[str]] = None
) -> pd.DataFrame:

    docs = docs.map_pipeline(nlp)

    if torch.cuda.is_available():
        print("Using GPU")
        docs = docs.set_processing(
            num_cpu_workers=4,
            num_gpu_workers=1,
            batch_size=32,
        )

    def converter_with_attributes(doc: Doc):
        return convert_doc_to_dict(doc, attributes=attributes)

    df_ents = edsnlp.data.to_pandas( # type: ignore
        docs,
        converter=converter_with_attributes,
    )

    return df_ents

def main(
    root: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    try:
        from biomedics.ner.loaders import eds_biomedic
        if torch.cuda.is_available():
            print("Using GPU")
            spacy.require_gpu() # type: ignore

        print("Using EDS-Biomedic")
        nlp = eds_biomedic()
    except ImportError:
        sys.exit("EDS-Biomedic not found, please define the way to load the model.")

    docs = build_data(root)
    basic_attributes = [
        "Negation",
        "Certainty",
        "Family",
        "Action",
        "Temporality",
    ]
    df_ents = extract_ents_from_docs(docs, nlp, basic_attributes)
    df_ents[basic_attributes] = (
        df_ents[basic_attributes]
        .fillna(False)
        .replace('nan', False)
        .astype(bool)
    )
    if output_path:
        df_ents.to_parquet(
            output_path,
            engine="pyarrow",
            compression="snappy"
        )

    return df_ents

if __name__ == "__main__":
    typer.run(main)
