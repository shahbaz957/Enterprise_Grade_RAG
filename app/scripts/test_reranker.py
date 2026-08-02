from app.services.retrieval import search_enterprise_knowledge

hits = search_enterprise_knowledge("How do Kubernetes Jobs process a work queue?", limit=15)
for h in hits:
    print(h.rank, h.score, h.metadata["filename"], h.text[:200])