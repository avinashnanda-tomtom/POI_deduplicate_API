import numpy as np


def name_distance(model,name1, name2, batch=300000):
    sims = np.empty((0), np.float32)
    for i in range(0, len(name1), batch):
        embeddings1 = model.encode(name1[i:i + batch],
                                   batch_size=64,
                                   normalize_embeddings=True)
        embeddings2 = model.encode(name2[i:i + batch],
                                   batch_size=64,
                                   normalize_embeddings=True)
        cosine = np.sum(embeddings1 * embeddings2, axis=1)
        cosine = np.round(cosine, 3)
        sims = np.concatenate((sims, cosine))
    return sims