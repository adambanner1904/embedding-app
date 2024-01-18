import umap.umap_ as umap


def use_umap(data, metric='cosine'):
    points = umap.UMAP(metric='cosine', random_state=42).fit_transform(data)
    return points

