"""News article clustering using TF-IDF, K-Means and hierarchical clustering.
Run from repository root: python news_clustering/main.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import Normalizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'news'
OUT.mkdir(parents=True, exist_ok=True)


def metrics(x, labels):
    return {'silhouette': silhouette_score(x, labels),
            'davies_bouldin': davies_bouldin_score(x, labels),
            'calinski_harabasz': calinski_harabasz_score(x, labels)}


def main():
    df = pd.read_csv(ROOT / 'data' / 'news_articles.csv')
    df['text'] = df['text'].fillna('').astype(str).str.strip()
    df = df.drop_duplicates(subset=['text'])
    df = df[df.text.str.len() >= 15].copy().reset_index(drop=True)
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=1, ngram_range=(1, 2))
    sparse = tfidf.fit_transform(df.text)
    # Reduce sparse text to dense components; normalize for clustering and plots.
    n_components = min(30, sparse.shape[0] - 1, sparse.shape[1] - 1)
    if n_components < 2:
        raise ValueError('Need more distinct articles and vocabulary for clustering')
    x = Normalizer().fit_transform(TruncatedSVD(n_components=n_components, random_state=42).fit_transform(sparse))
    ks = range(2, min(9, len(df)))
    selection = []
    for k in ks:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(x)
        selection.append({'k': k, 'silhouette': silhouette_score(x, labels)})
    selection = pd.DataFrame(selection)
    k = int(selection.loc[selection.silhouette.idxmax(), 'k'])
    df['kmeans_cluster'] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(x)
    df['hierarchical_cluster'] = AgglomerativeClustering(n_clusters=k, linkage='average', metric='cosine').fit_predict(x)
    pd.DataFrame([{'method': name, 'k': k, **metrics(x, df[column])} for name, column in
                  [('K-Means', 'kmeans_cluster'), ('Hierarchical (cosine)', 'hierarchical_cluster')]]).to_csv(OUT / 'evaluation.csv', index=False)
    selection.to_csv(OUT / 'k_selection.csv', index=False)
    df.to_csv(OUT / 'clustered_articles.csv', index=False)
    # Top weighted TF-IDF terms for each cluster: interpretable article topics.
    vocab = np.array(tfidf.get_feature_names_out())
    with (OUT / 'cluster_keywords.txt').open('w', encoding='utf-8') as f:
        for group in sorted(df.kmeans_cluster.unique()):
            avg = np.asarray(sparse[df.index[df.kmeans_cluster == group]].mean(axis=0)).ravel()
            terms = ', '.join(vocab[np.argsort(avg)[-8:][::-1]])
            f.write(f'Cluster {group} ({sum(df.kmeans_cluster == group)} articles): {terms}\n')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(selection.k, selection.silhouette, marker='o')
    ax.set(title='Select K using silhouette', xlabel='Number of clusters', ylabel='Silhouette')
    fig.tight_layout(); fig.savefig(OUT / 'k_selection.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, name in zip(axes, ['kmeans_cluster', 'hierarchical_cluster'], ['K-Means', 'Hierarchical']):
        ax.scatter(x[:, 0], x[:, 1], c=df[col], cmap='tab10', s=50)
        ax.set(title=name + ' (first two SVD components)', xlabel='Component 1', ylabel='Component 2')
    fig.tight_layout(); fig.savefig(OUT / 'article_clusters.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(linkage(cosine_distances(x)[np.triu_indices(len(x), k=1)], method='average'),
               ax=ax, no_labels=True)
    ax.set(title='Hierarchical clustering dendrogram', ylabel='Cosine distance')
    fig.tight_layout(); fig.savefig(OUT / 'dendrogram.png', dpi=160); plt.close(fig)
    print(f'Processed {len(df)} articles; selected k={k}. Results: {OUT}')
    print(pd.read_csv(OUT / 'evaluation.csv').to_string(index=False))
    print((OUT / 'cluster_keywords.txt').read_text(encoding='utf-8'))

if __name__ == '__main__':
    main()
