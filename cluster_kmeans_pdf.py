## Install the necessary libraries: pip install PyPDF2 pandas scikit-learn pdfplumber matplotlib seaborn openpyxl requests plotly
import os
import pdfplumber
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import mean

def extract_text_from_single_pdf(file_path):
    """Extract text from a PDF. Optimized with pdfplumber."""
    try:
        with pdfplumber.open(file_path) as pdf:
            pages_text = [page.extract_text() for page in pdf.pages]
            text = "\n".join(pages_text)
            return text if text.strip() else None
    except Exception as e:
        print(f"Error in processing {file_path}: {e}")
        return None

def extract_text_from_pdfs(folder_path, use_parallel=True, max_workers=4):
    #Extract text from PDFs

    pdf_files = list(Path(folder_path).glob('*.pdf'))
    
    if not pdf_files:
        print("No PDF files found in the specified folder.")
        return []
    
    if use_parallel and len(pdf_files) > 1:
        texts = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(extract_text_from_single_pdf, pdf_files)
            texts = [text for text in results if text is not None]
    else:
        texts = []
        for file_path in pdf_files:
            text = extract_text_from_single_pdf(file_path)
            if text is not None:
                texts.append(text)
    
    return texts

def process_and_cluster_pdfs(folder_path, n_clusters=5, use_parallel=True):
    # Step 1: Extract text from PDFs (optimized)
    texts = extract_text_from_pdfs(folder_path, use_parallel=use_parallel)
    
    if not texts:
        print("No PDF files found in the specified folder.")
        return pd.DataFrame()

    vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_df=0.8, 
        min_df=2,
        max_features=1000  # limiting features for better performance
    )
    X_tfidf = vectorizer.fit_transform(texts)

    # Data scaling (important for K-Means)
    scaler = StandardScaler(with_mean=False)
    X_scaled = hstack([X_tfidf, scaler.fit_transform(X_tfidf)])

    # Clustering with K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clusters = pd.DataFrame({
        'text': texts,
        'cluster': kmeans.fit_predict(X_scaled),
    })

    return df_clusters


def get_cluster_stats(df_clusters):
    # Cluster statistics: quantity and percentage of samples in each cluster
    stats = df_clusters.groupby('cluster').size().reset_index(name='quantity')
    stats['percentage'] = (stats['quantity'] / len(df_clusters) * 100).round(2)
    return stats


def get_top_keywords_per_cluster(texts, clusters, vectorizer, n_keywords=5):
    # KRepresentative keywords for each cluster
    terms = vectorizer.get_feature_names_out()
    clusters_arr = clusters.values if hasattr(clusters, 'values') else clusters
    keywords_by_cluster = {}
    
    for cluster_id in set(clusters_arr):
        mask = clusters_arr == cluster_id
        cluster_texts = [texts[i] for i in range(len(texts)) if mask[i]]
        
        if cluster_texts:
            cluster_vect = vectorizer.transform(cluster_texts).mean(axis=0).A1
            top_indices = cluster_vect.argsort()[-n_keywords:][::-1]
            keywords_by_cluster[cluster_id] = [terms[i] for i in top_indices]
    
    return keywords_by_cluster


def evaluate_clustering(X_scaled, clusters, vectorizer):
    # Quality metrics for clustering: Silhouette Score and Davies-Bouldin Index
    silhouette = silhouette_score(X_scaled, clusters)
    davies_bouldin = davies_bouldin_score(X_scaled, clusters)
    
    metrics = {
        'silhouette_score': round(silhouette, 4),
        'davies_bouldin_score': round(davies_bouldin, 4),
        'info': f"Silhouette: {silhouette:.4f} (cercano a 1 es mejor) | DB: {davies_bouldin:.4f} (menor es mejor)"
    }
    return metrics


def export_results(df_clusters, stats, keywords, output_file='cluster_results.xlsx'):
    # Excel export with clusters, statistics, and keywords
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_clusters.to_excel(writer, sheet_name='Clusters', index=False)
        stats.to_excel(writer, sheet_name='Statistics', index=False)
    print(f"Results exported to {output_file}")


def plot_cluster_distribution(df_clusters, title="Cluster distribution"):
    # Graph bar of cluster distribution
    plt.figure(figsize=(10, 5))
    df_clusters['cluster'].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Cluster')
    plt.ylabel('Quantity of PDFs')
    plt.tight_layout()
    plt.show()


def plot_cluster_size_pie(df_clusters, title="Cluster size distribution"):
    # Graph pie of cluster size distribution
    plt.figure(figsize=(8, 8))
    plt.pie(df_clusters['cluster'].value_counts().sort_index(), 
            labels=[f"Cluster {i}" for i in sorted(df_clusters['cluster'].unique())],
            autopct='%1.1f%%', startangle=90)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def get_cluster_summary(df_clusters, keywords_dict):
    # Cluster summary with quantity of documents and representative keywords
    summary = []
    for cluster_id in sorted(df_clusters['cluster'].unique()):
        n_docs = len(df_clusters[df_clusters['cluster'] == cluster_id])
        keywords = ', '.join(keywords_dict.get(cluster_id, [])[:3])
        summary.append({
            'cluster': cluster_id,
            'documents': n_docs,
            'keywords': keywords
        })
    return pd.DataFrame(summary)


if __name__ == "__main__":
    folder_path_to_pdfs = input("Insert the path to the folder containing the PDFs: ")
    n_clusters = int(input("NNumber of clusters (default 3): ") or 3)
    
    # Extract text from PDFs
    texts = extract_text_from_pdfs(folder_path_to_pdfs, use_parallel=True)
    if not texts:
        exit()
    
    # Vectorization and scaling
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=2, max_features=1000)
    X_tfidf = vectorizer.fit_transform(texts)
    scaler = StandardScaler(with_mean=False)
    X_scaled = hstack([X_tfidf, scaler.fit_transform(X_tfidf)])
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df_clusters = pd.DataFrame({'text': texts, 'cluster': clusters})
    
    # Analysis of results
    stats = get_cluster_stats(df_clusters)
    keywords = get_top_keywords_per_cluster(texts, pd.Series(clusters), vectorizer)
    metrics = evaluate_clustering(X_scaled, clusters, vectorizer)
    summary = get_cluster_summary(df_clusters, keywords)
    
    # Results presentation
    print("\n" + "="*60)
    print("CLUSTERING STATISTICS")
    print("="*60)
    print(stats.to_string(index=False))
    print("\n" + "="*60)
    print("CLUSTER SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))
    print("\n" + "="*60)
    print("QUALITY METRICS")
    print("="*60)
    print(metrics['info'])
    
    # Export results and visualizations
    export_results(df_clusters, stats, keywords)
    plot_cluster_distribution(df_clusters)
    plot_cluster_size_pie(df_clusters)