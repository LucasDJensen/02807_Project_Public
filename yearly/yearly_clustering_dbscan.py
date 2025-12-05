import os
from pathlib import Path

import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

from _config import *

cmap = "tab20"

out_path = Path(r'C:\Users\lucas\PycharmProjects\02807_Project\output\data\year_clustering_dbscan_elbow')
out_path.mkdir(parents=True, exist_ok=True)


def elbow_silhouette_davies(df: pd.DataFrame) -> float:
    """
    DBSCAN variant:
    - Sweep over eps (with fixed min_samples)
    - Plot number of clusters, silhouette and Davies–Bouldin vs eps
    """

    # ---- parameters to scan ----
    eps_values = np.linspace(0.5, 10.0, 100)  # tweak as needed
    default_min_samples = 5                 # tweak as needed

    n_clusters_list = []
    silhouette_scores = []
    dbi_scores = []

    X = df.values

    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=default_min_samples)
        labels = dbscan.fit_predict(X)

        # Number of clusters (excluding noise = -1)
        unique_labels = set(labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        n_clusters_list.append(n_clusters)

        # If no valid clusters or only 1 cluster, silhouette/DBI are not defined
        if n_clusters <= 1:
            silhouette_scores.append(np.nan)
            dbi_scores.append(np.nan)
            continue

        # Evaluate only on non-noise points
        mask = labels != -1
        X_core = X[mask]
        labels_core = labels[mask]

        silhouette_scores.append(silhouette_score(X_core, labels_core))
        dbi_scores.append(davies_bouldin_score(X_core, labels_core))
    # argmax of eps from n_clusters_list
    idx = np.argmax(n_clusters_list)
    print(f"Optimal eps: {eps_values[idx]:.2f}")
    return eps_values[idx]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # Number of clusters vs eps
    axes[0].plot(eps_values, n_clusters_list, marker='o')
    axes[0].set_title('DBSCAN: Number of clusters vs. eps')
    axes[0].set_xlabel('eps')
    axes[0].set_ylabel('Number of clusters')
    axes[0].grid(True)
    axes[0].set_xlim(0, 10)
    # ticks
    axes[0].xaxis.set_major_locator(plt.MultipleLocator(0.25))
    # rotate
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")

    # Silhouette vs eps
    axes[1].plot(eps_values, silhouette_scores, marker='o')
    axes[1].set_title('DBSCAN: Silhouette Score vs. eps (higher is better)')
    axes[1].set_xlabel('eps')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].grid(True)

    # Davies–Bouldin vs eps
    axes[2].plot(eps_values, dbi_scores, marker='o')
    axes[2].set_title('DBSCAN: Davies–Bouldin Index vs. eps (lower is better)')
    axes[2].set_xlabel('eps')
    axes[2].set_ylabel('Davies–Bouldin Index')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


def relabel_dbscan_labels(labels: np.ndarray) -> pd.Series:
    """
    Keep noise as -1, reindex actual clusters by size:
    largest -> 0, second largest -> 1, ...
    """
    labels_series = pd.Series(labels)

    # Separate noise and clusters
    noise_mask = labels_series == -1
    cluster_labels = labels_series[~noise_mask]

    if cluster_labels.empty:
        # all noise, nothing to relabel
        return labels_series

    order = cluster_labels.value_counts().index   # largest first
    relabel_map = {old: new for new, old in enumerate(order)}

    # Apply mapping to cluster labels, keep noise as -1
    labels_reindexed = labels_series.copy()
    for old, new in relabel_map.items():
        labels_reindexed[labels_series == old] = new

    return labels_reindexed


complete_years = pd.read_csv(
    PATH_PROJECT / r'output\data\discharge_tables\complete_years.csv',
    index_col=0,
    dtype=str
)

# for year in [2007, 2008]:
for year in complete_years.index:
    print(year)
    df = pd.read_csv(
        PATH_PROJECT / r'output\data\discharge_tables\discharge_table_2001_2022.csv',
        index_col=0,
        parse_dates=True
    )

    cids = complete_years.loc[year]
    cids = cids[~cids.isna()].values
    year_df = df.loc[str(year)][cids]

    year_df_scaled = ((year_df - year_df.mean()) / year_df.std(ddof=0)).copy()
    year_df_scaled = year_df_scaled.dropna(axis=1)

    # ---- DBSCAN parameter diagnostics on raw scaled data ----
    eps = elbow_silhouette_davies(year_df_scaled.T.copy())

    # input DBSCAN parameters
    # eps = float(input("Enter eps for DBSCAN (raw data): "))
    # min_samples = int(input("Enter min_samples for DBSCAN (raw data): "))
    min_samples = 5

    # Apply DBSCAN clustering
    dbscan_final = DBSCAN(eps=eps, min_samples=min_samples)
    labels_final = dbscan_final.fit_predict(year_df_scaled.T)

    # Reindex clusters (keep noise = -1)
    labels_reindexed = relabel_dbscan_labels(labels_final)
    labels_series = pd.Series(labels_reindexed.values, index=year_df_scaled.T.index)

    # Assign cluster labels to each catchment
    clustered_data = pd.DataFrame(index=year_df_scaled.T.index)
    clustered_data['Cluster'] = labels_series
    clustered_data.sort_values(by='Cluster', inplace=True)

    file = out_path / f'DBSCAN_clustering_year_{str(year)}.csv'
    clustered_data.to_csv(file, index=True)

    cluster_file = str(file)
    station_mapping = pd.read_csv(os.getenv('STATION_MAPPING'), dtype=str)
    shape_file = Path(os.getenv('SHAPE_FILE'))
    all_catchments_gdf = geopandas.read_file(shape_file)
    cluster_mapping = pd.read_csv(cluster_file, dtype=str)
    cluster_mapping.columns = ['id', 'cluster']

    new_column = []
    for catchment in cluster_mapping['id'].values:
        new_column.append(
            np.int64(
                station_mapping[station_mapping["obsstednr"] == catchment]["Id15_v30"].values[0]
            )
        )

    cluster_mapping['Id15_v30'] = new_column
    cluster_mapping.drop(columns=['id'], inplace=True)

    # filter out gauged catchments
    idxs = all_catchments_gdf['Id15_v30'].isin(
        [np.int64(x) for x in cluster_mapping['Id15_v30'].values]
    )
    gauged_catchments_gdf = all_catchments_gdf[idxs]
    ungauged_catchments_gdf = all_catchments_gdf[idxs == False]
    merged_gdf = gauged_catchments_gdf.merge(cluster_mapping, on='Id15_v30')
    merged_gdf['area'] = merged_gdf.geometry.area

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Plot without automatic legend
    merged_gdf.sort_values(by="area", ascending=False).plot(
        column="cluster",
        figsize=(25, 15),
        ax=ax,
        legend=False,
        cmap=cmap
    )

    ax.set_title(f"{year} DBSCAN clustering")

    # Add Denmark boundary
    denmark_boundary = all_catchments_gdf.boundary
    denmark_boundary.plot(ax=ax, color='black', linewidth=0.02)

    # Axis settings
    ax.set_xlim(left=440000, right=730000)
    ax.set_ylim(top=6.4e6, bottom=6.05e6)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # --------------------------------------------------------
    #  CUSTOM LEGEND (2 columns, small size, upper-right)
    # --------------------------------------------------------
    import matplotlib as mpl
    import matplotlib.patches as mpatches

    cluster_cat = merged_gdf["cluster"].astype("category")
    clusters_ordered = list(cluster_cat.cat.categories)

    cmap_obj = mpl.colormaps[cmap]

    n_clusters = len(clusters_ordered)

    handles = []
    for i, c in enumerate(clusters_ordered):
        if n_clusters == 1:
            t = 0.5
        else:
            t = i / (n_clusters - 1)
        color = cmap_obj(t)

        handles.append(
            mpatches.Patch(
                color=color,
                label=str(c)
            )
        )

    ax.legend(
        handles=handles,
        title="Clusters",
        loc="upper right",
        bbox_to_anchor=(1, 1),
        ncol=2,
        fontsize=6,
        title_fontsize=7,
        frameon=True,
        borderpad=0.3,
        handlelength=1.0
    )

    plt.tight_layout()
    plt.savefig(
        str(Path(cluster_file).parent / Path(cluster_file).stem) + '.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()

    # ===================== PCA PART =====================

    df = year_df.copy()
    df_scaled = (df - df.mean()) / df.std(ddof=0)
    df_scaled = df_scaled.dropna(axis=1)

    # Compute cumulative explained variance for PCA components ranging from 1 to 50
    max_components = 50
    pca_full = PCA(n_components=max_components)
    pca_full.fit(df_scaled)

    explained_variance = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Plot cumulative explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_components + 1), cumulative_variance, marker='o')
    plt.title("Cumulative Explained Variance by PCA Components")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plt.axhline(
        y=0.9,
        color='r',
        linestyle='--',
        label=f'90% Variance. Components: {sum(~(cumulative_variance > 0.90))}'
    )
    plt.axhline(
        y=0.95,
        color='g',
        linestyle='--',
        label=f'95% Variance. Components: {sum(~(cumulative_variance > 0.95))}'
    )
    plt.axhline(
        y=0.98,
        color='y',
        linestyle='--',
        label=f'98% Variance. Components: {sum(~(cumulative_variance > 0.98))}'
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    # select number of components based on cumulative explained variance plot at 95%
    n_components = sum(~(cumulative_variance > 0.95)) + 1
    print(f"Number of PCA components: {n_components}")
    # input number of components
    # n_components = int(input("Enter number of components: "))
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df_scaled.T)
    pca_result = pd.DataFrame(
        pca_result,
        columns=[f'PCA{x}' for x in range(1, n_components + 1)],
        index=df_scaled.T.index
    )

    # Transpose the data to have catchments as rows and PCA features as columns
    data = pca_result.copy()

    # Normalize
    data_scaled = (data - data.mean()) / data.std(ddof=0)
    data_scaled = data_scaled.dropna(axis=1)

    # ---- DBSCAN parameter diagnostics on PCA features ----
    eps_pca = elbow_silhouette_davies(data_scaled.copy())

    # eps_pca = float(input("Enter eps for DBSCAN (PCA data): "))
    # min_samples_pca = int(input("Enter min_samples for DBSCAN (PCA data): "))
    min_samples_pca = 5

    dbscan_final_pca = DBSCAN(eps=eps_pca, min_samples=min_samples_pca)
    labels_final_pca = dbscan_final_pca.fit_predict(data_scaled)

    labels_reindexed_pca = relabel_dbscan_labels(labels_final_pca)
    labels_series_pca = pd.Series(labels_reindexed_pca.values, index=data_scaled.index)

    clustered_data_pca = pd.DataFrame(index=data_scaled.index)
    clustered_data_pca['Cluster'] = labels_series_pca
    clustered_data_pca.sort_values(by='Cluster', inplace=True)

    file = out_path / f'DBSCAN_clustering_year_pca_{str(year)}.csv'
    clustered_data_pca.to_csv(file, index=True)

    cluster_file = str(file)
    station_mapping = pd.read_csv(os.getenv('STATION_MAPPING'), dtype=str)
    shape_file = Path(os.getenv('SHAPE_FILE'))
    all_catchments_gdf = geopandas.read_file(shape_file)
    cluster_mapping = pd.read_csv(cluster_file, dtype=str)
    cluster_mapping.columns = ['id', 'cluster']

    new_column = []
    for catchment in cluster_mapping['id'].values:
        new_column.append(
            np.int64(
                station_mapping[station_mapping["obsstednr"] == catchment]["Id15_v30"].values[0]
            )
        )

    cluster_mapping['Id15_v30'] = new_column
    cluster_mapping.drop(columns=['id'], inplace=True)

    idxs = all_catchments_gdf['Id15_v30'].isin(
        [np.int64(x) for x in cluster_mapping['Id15_v30'].values]
    )
    gauged_catchments_gdf = all_catchments_gdf[idxs]
    ungauged_catchments_gdf = all_catchments_gdf[idxs == False]
    merged_gdf = gauged_catchments_gdf.merge(cluster_mapping, on='Id15_v30')
    merged_gdf['area'] = merged_gdf.geometry.area

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    merged_gdf.sort_values(by="area", ascending=False).plot(
        column="cluster",
        figsize=(25, 15),
        ax=ax,
        legend=False,
        cmap=cmap
    )

    ax.set_title(f"{year} DBSCAN clustering PCA")

    denmark_boundary = all_catchments_gdf.boundary
    denmark_boundary.plot(ax=ax, color='black', linewidth=0.02)

    ax.set_xlim(left=440000, right=730000)
    ax.set_ylim(top=6.4e6, bottom=6.05e6)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    import matplotlib as mpl
    import matplotlib.patches as mpatches

    cluster_cat = merged_gdf["cluster"].astype("category")
    clusters_ordered = list(cluster_cat.cat.categories)

    cmap_obj = mpl.colormaps[cmap]
    n_clusters = len(clusters_ordered)

    handles = []
    for i, c in enumerate(clusters_ordered):
        if n_clusters == 1:
            t = 0.5
        else:
            t = i / (n_clusters - 1)
        color = cmap_obj(t)

        handles.append(
            mpatches.Patch(
                color=color,
                label=str(c)
            )
        )

    ax.legend(
        handles=handles,
        title="Clusters",
        loc="upper right",
        bbox_to_anchor=(1, 1),
        ncol=2,
        fontsize=6,
        title_fontsize=7,
        frameon=True,
        borderpad=0.3,
        handlelength=1.0
    )

    plt.tight_layout()
    plt.savefig(
        str(Path(cluster_file).parent / Path(cluster_file).stem) + '.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()
