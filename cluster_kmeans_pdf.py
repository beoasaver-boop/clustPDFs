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

def validate_clustering_inputs(num_pdfs, n_clusters):
    """Validate and adjust clustering inputs for small collections.
    
    Args:
        num_pdfs: Number of PDF documents
        n_clusters: Requested number of clusters
    
    Returns:
        Adjusted n_clusters value valid for the number of PDFs
    """
    if num_pdfs < 1:
        raise ValueError("At least 1 PDF is required for clustering")
    
    if n_clusters is None or n_clusters < 1:
        n_clusters = min(3, num_pdfs)
    elif n_clusters > num_pdfs:
        print(f"⚠️  Warning: Cannot create {n_clusters} clusters from {num_pdfs} PDF(s)")
        n_clusters = num_pdfs
        print(f"   Adjusted to {n_clusters} cluster(s)")
    
    return n_clusters


def get_optimal_vectorizer_params(num_pdfs):
    """Get optimal TfidfVectorizer parameters based on number of documents.
    
    Args:
        num_pdfs: Number of PDF documents
    
    Returns:
        Dictionary with vectorizer parameters
    """
    if num_pdfs == 1:
        return {'min_df': 1, 'max_df': 1.0}
    elif num_pdfs < 5:
        return {'min_df': 1, 'max_df': 0.95}
    else:
        return {'min_df': 2, 'max_df': 0.8}


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
    """Extract text from PDFs. Supports small collections (minimum 1 PDF)."""

    pdf_files = list(Path(folder_path).glob('*.pdf'))
    
    if not pdf_files:
        print("No PDF files found in the specified folder.")
        return []
    
    print(f"✓ Found {len(pdf_files)} PDF file(s)")
    
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

def process_and_cluster_pdfs(folder_path, n_clusters=None, use_parallel=True):
    """Process and cluster PDFs with support for small collections (minimum 1 PDF).
    
    Args:
        folder_path: Path to folder with PDFs
        n_clusters: Number of clusters (auto-determined if None or invalid)
        use_parallel: Whether to use parallel processing
    
    Returns:
        DataFrame with clustered texts or empty DataFrame if no PDFs found
    """
    # Step 1: Extract text from PDFs (optimized)
    texts = extract_text_from_pdfs(folder_path, use_parallel=use_parallel)
    
    if not texts:
        print("❌ No PDF files found in the specified folder.")
        return pd.DataFrame()
    
    num_pdfs = len(texts)
    print(f"\n📄 Processing {num_pdfs} PDF(s)...")
    
    # Validate and adjust n_clusters for small collections
    if n_clusters is None or n_clusters < 1:
        n_clusters = min(3, num_pdfs)  # Default to 3, but max is num_pdfs
    elif n_clusters > num_pdfs:
        print(f"⚠️  Warning: n_clusters ({n_clusters}) cannot exceed number of PDFs ({num_pdfs})")
        n_clusters = num_pdfs
        print(f"   Adjusted n_clusters to {n_clusters}")
    
    # Dynamically adjust min_df based on number of documents
    # For single PDF: min_df=1, for small collections: min_df=1, for larger: min_df=2
    min_df = 1 if num_pdfs < 5 else 2
    max_df = 0.9 if num_pdfs < 5 else 0.8
    
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_df=max_df, 
        min_df=min_df,
        max_features=1000  # limiting features for better performance
    )
    
    try:
        X_tfidf = vectorizer.fit_transform(texts)
    except ValueError as e:
        print(f"❌ Error during vectorization: {e}")
        print("   This might be due to insufficient text or vocabulary.")
        return pd.DataFrame()

    # Data scaling (important for K-Means)
    scaler = StandardScaler(with_mean=False)
    X_scaled = hstack([X_tfidf, scaler.fit_transform(X_tfidf)])

    # Clustering with K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clusters = pd.DataFrame({
        'text': texts,
        'cluster': kmeans.fit_predict(X_scaled),
    })
    
    print(f"✓ Clustering complete: {n_clusters} cluster(s) created")

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
    # Convert sparse matrix to dense array for sklearn metrics
    if hasattr(X_scaled, 'toarray'):
        X_dense = X_scaled.toarray()
    else:
        X_dense = X_scaled
    
    # Quality metrics for clustering: Silhouette Score and Davies-Bouldin Index
    silhouette = silhouette_score(X_dense, clusters)
    davies_bouldin = davies_bouldin_score(X_dense, clusters)
    
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
    print("="*60)
    print("PDF CLUSTERING WITH K-MEANS")
    print("Supports small collections (minimum 1 PDF)")
    print("="*60 + "\n")
    
    folder_path_to_pdfs = input("Insert the path to the folder containing the PDFs: ").strip()
    
    # Validate folder path
    if not os.path.isdir(folder_path_to_pdfs):
        print(f"❌ Error: Folder '{folder_path_to_pdfs}' does not exist.")
        exit(1)
    
    # Get number of PDFs
    pdf_files = list(Path(folder_path_to_pdfs).glob('*.pdf'))
    if not pdf_files:
        print(f"❌ Error: No PDF files found in '{folder_path_to_pdfs}'")
        exit(1)
    
    num_pdfs = len(pdf_files)
    
    # Get number of clusters with smart defaults
    default_clusters = min(3, num_pdfs)
    clusters_input = input(f"Number of clusters (default {default_clusters}, max {num_pdfs}): ").strip()
    
    try:
        n_clusters = int(clusters_input) if clusters_input else default_clusters
    except ValueError:
        print(f"⚠️  Invalid input. Using default: {default_clusters} clusters")
        n_clusters = default_clusters
    
    # Extract text from PDFs
    texts = extract_text_from_pdfs(folder_path_to_pdfs, use_parallel=True)
    if not texts:
        exit(1)
    
    # Process and cluster
    df_clusters = process_and_cluster_pdfs(folder_path_to_pdfs, n_clusters=n_clusters, use_parallel=True)
    
    if df_clusters.empty:
        print("❌ Failed to process PDFs.")
        exit(1)
    
    # Vectorization for additional analysis
    min_df = 1 if num_pdfs < 5 else 2
    max_df = 0.9 if num_pdfs < 5 else 0.8
    vectorizer = TfidfVectorizer(stop_words='english', max_df=max_df, min_df=min_df, max_features=1000)
    X_tfidf = vectorizer.fit_transform(texts)
    scaler = StandardScaler(with_mean=False)
    X_scaled = hstack([X_tfidf, scaler.fit_transform(X_tfidf)])
    
    # Clustering
    clusters = df_clusters['cluster'].values
    
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