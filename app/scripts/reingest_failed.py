# scripts/reingest_failed.py
from app.ingestion.processor import process_file

FAILED_TRUE = [
    "pods_autoscale.html",
]

FAILED_NOISY = [
    "2018 CppCon Unwinding the Stack - Exploring how C++ Exceptions work on Windows - James McNellis.pdf",
    "5-Level Paging and 5-Level EPT - Intel - Revision 1.0 (December, 2016).pdf",
    "5-Level Paging and 5-Level EPT - Intel - Revision 1.1 (May, 2017).pdf",
    "A Brief Introduction to Neural Networks (neuronalenetze-en-zeta2-2col-dkrieselcom).pdf",
    "A Forensic Analysis of CSG 11 Encounter with an Anomalous Aerial Vehicle.pdf",
    "A Journey in Creating an Operating System Kernel - The 539Kernel Book (Nov 2022).pdf",
    "A Mathematical Theory of Communication (1948).pdf",
    "A Nanopass Framework for Compiler Education.pdf",
    "A Neural Probabilistic Language Model (bengio03a).pdf",
    "A Novel Hybrid Quicksort Algorithm Vectorized using AVX-512 on Intel Skylake (1704.08579).pdf",
    "A PlusCal User's Manual - C-Syntax Version 1.8 (31 Aug 2018).pdf",
    "A Primer on Memory Consistency and Cache Coherence - 2nd Edition.pdf",
]

for name in FAILED_TRUE:
    path = f"DATA/true_data/{name}"
    result = process_file(path, corpus="true")
    print(f"[{'OK' if result.ok else 'ERR'}] {path} — {result.error or f'chunks={result.chunks}'}")

for name in FAILED_NOISY:
    path = f"DATA/noisy_data/{name}"
    result = process_file(path, corpus="noisy")
    print(f"[{'OK' if result.ok else 'ERR'}] {path} — {result.error or f'chunks={result.chunks}'}")