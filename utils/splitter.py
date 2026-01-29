from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    
    # Fix: Update metadata without overwriting original
    for doc in split_docs:
        doc.metadata["source_file"] = doc.metadata.get("source_file", "unknown")
        doc.metadata["file_type"] = doc.metadata.get("file_type", "pdf")
        doc.metadata["page"] = doc.metadata.get("page", -1)
    
    return split_docs

# Remove or comment out test code for module use
# if __name__ == "__main__":
#     chunks = split_documents(all_pdf_documents)
