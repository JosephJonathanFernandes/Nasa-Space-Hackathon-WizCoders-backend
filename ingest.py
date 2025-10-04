import argparse
import glob
import os
from retriever import Retriever


def ingest_folder(folder: str):
    r = Retriever()
    files = glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True)
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        r.add_documents_from_string(text, source=os.path.basename(p))
        print(f"Ingested {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing .txt files to ingest")
    args = parser.parse_args()
    ingest_folder(args.folder)
