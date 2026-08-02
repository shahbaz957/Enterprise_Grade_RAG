from app.ingestion.loaders import load_document

samples = [
    "DATA/true_data/parallel_work_queue.txt",
    "DATA/true_data/pods_autoscale.html",
    "DATA/true_data/cronjobs.docx",
    "DATA/true_data/architecture.pptx",
    "DATA/noisy_data/A New Algorithm for Data Compression (1994).pdf",
]

for path in samples:
    doc = load_document(path)
    name = doc.metadata["filename"]
    preview = doc.text[:120].replace("\n", " ").encode("ascii", "replace").decode("ascii")
    print(f"{doc.doc_type:5} chars={doc.char_count:6} empty={doc.is_empty} file={name}")
    print(f"  preview: {preview!r}")
    print()
