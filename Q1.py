"""Customer segmentation using K-Means and Agglomerative Clustering.
Run from repository root: python customer_segmentation/main.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'customer'
OUT.mkdir(parents=True, exist_ok=True)


def evaluate(x, labels):
    if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
        return {'silhouette': np.nan, 'davies_bouldin': np.nan, 'calinski_harabasz': np.nan}
    return {'silhouette': silhouette_score(x, labels),
            'davies_bouldin': davies_bouldin_score(x, labels),
            'calinski_harabasz': calinski_harabasz_score(x, labels)}


def main():
    df = pd.read_csv(ROOT / 'data' / 'customers.csv')
    features = ['age', 'annual_income_k', 'spending_score', 'purchases_per_month']
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')
    df = df.drop_duplicates(subset=['customer_id']).dropna(subset=features).copy()
    x = StandardScaler().fit_transform(df[features])
    ks = range(2, min(9, len(df)))
    scores = []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(x)
        scores.append({'k': k, 'inertia': model.inertia_, 'silhouette': silhouette_score(x, labels)})
    score_df = pd.DataFrame(scores)
    best_k = int(score_df.loc[score_df.silhouette.idxmax(), 'k'])
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['kmeans_cluster'] = km.fit_predict(x)
    df['hierarchical_cluster'] = AgglomerativeClustering(n_clusters=best_k, linkage='ward').fit_predict(x)
    pd.DataFrame([{'method': method, 'k': best_k, **evaluate(x, df[col])}
                  for method, col in [('K-Means', 'kmeans_cluster'),
                                      ('Hierarchical (Ward)', 'hierarchical_cluster')]]).to_csv(OUT / 'evaluation.csv', index=False)
    score_df.to_csv(OUT / 'k_selection.csv', index=False)
    df.to_csv(OUT / 'segmented_customers.csv', index=False)
    df.groupby('kmeans_cluster')[features].agg(['mean', 'count']).round(2).to_csv(OUT / 'cluster_profiles.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(score_df.k, score_df.inertia, marker='o')
    axes[0].set(title='Elbow method', xlabel='Number of clusters', ylabel='Inertia')
    axes[1].plot(score_df.k, score_df.silhouette, marker='o')
    axes[1].set(title='Silhouette score', xlabel='Number of clusters', ylabel='Silhouette')
    fig.tight_layout(); fig.savefig(OUT / 'cluster_selection.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in zip(axes, ['kmeans_cluster', 'hierarchical_cluster'], ['K-Means', 'Hierarchical']):
        sns.scatterplot(data=df, x='annual_income_k', y='spending_score', hue=col, palette='tab10', ax=ax)
        ax.set_title(title)
    fig.tight_layout(); fig.savefig(OUT / 'customer_clusters.png', dpi=160); plt.close(fig)
    print(f'Processed {len(df)} customers; selected k={best_k}. Results: {OUT}')
    print(pd.read_csv(OUT / 'evaluation.csv').to_string(index=False))
    print('Business interpretation: use cluster_profiles.csv to compare typical age, income, spending and purchase frequency.')

if __name__ == '__main__':
    main()
