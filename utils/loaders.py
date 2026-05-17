from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path


def process_all_pdfs(pdf_directory: str):
    """
    Load all PDF files from a directory recursively.

    Args:
        pdf_directory: Path to the directory containing PDFs
    Returns:
        List of LangChain Document objects
    """
    all_documents = []
    pdf_dir = Path(pdf_directory)

    if not pdf_dir.exists():
        print(f"Directory not found: {pdf_directory}")
        return all_documents

    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process")

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()

            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"

            all_documents.extend(documents)
            print(f"✓ Loaded {len(documents)} pages")
        except Exception as e:
            print(f"✗ Error loading {pdf_file.name}: {e}")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents


# ── No module-level execution ───────────────────────────────────────────────
# Previously: all_pdf_documents = process_all_pdfs("../data/documents")  ← ran on every import
# Now called only from ingest.py