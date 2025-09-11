import numpy as np
from sklearn.cluster import KMeans

def build_prototypes(embeddings: np.ndarray, k=64, random_state=0):
    if len(embeddings) < k: k = max(2, len(embeddings))
    km = KMeans(n_clusters=k, n_init=5, random_state=random_state).fit(embeddings)
    return km.cluster_centers_, km.labels_
