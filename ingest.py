import argparse
import glob
import os
from retriever import Retriever

try:
    import PyPDF2
except Exception:
    PyPDF2 = None


def _extract_text_from_pdf(path: str) -> str:
    if PyPDF2 is None:
        raise RuntimeError("PyPDF2 is not installed; install with: pip install PyPDF2")
    text_parts = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                continue
    return "\n".join(text_parts)


def ingest_folder(folder: str):
    r = Retriever()
    # gather txt and pdf files
    txt_files = glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True)
    pdf_files = glob.glob(os.path.join(folder, "**", "*.pdf"), recursive=True)

    for p in txt_files:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        r.add_documents_from_string(text, source=os.path.basename(p))
        print(f"Ingested {p}")

    for p in pdf_files:
        try:
            text = _extract_text_from_pdf(p)
            if text.strip():
                r.add_documents_from_string(text, source=os.path.basename(p))
                print(f"Ingested {p}")
            else:
                print(f"Warning: No text extracted from {p}")
        except Exception as e:
            print(f"Error extracting {p}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing .txt/.pdf files to ingest")
    args = parser.parse_args()
    ingest_folder(args.folder)
