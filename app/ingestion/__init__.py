"""Ingestion package — loaders, chunking, processor."""

# Keep this module light so `python -m app.ingestion.processor` doesn't warn.
# Import from submodules directly:
#   from app.ingestion.processor import process_file, ...
#   from app.ingestion.loaders import load_document
#   from app.ingestion.chunking import split_document
