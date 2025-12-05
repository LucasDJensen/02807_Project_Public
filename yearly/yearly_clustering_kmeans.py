import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

from _config import *

cmap = "tab20"

out_path = Path(r'C:\Users\lucas\PycharmProjects\02807_Project\output\data\year_clustering_elbow')
out_path.mkdir(parents=True, exist_ok=True)


def elbow_silhouette_davies(df:pd.DataFrame) -> None:

    # Try different numbers of clusters and calculate the inertia, silhouette, and Davies–Bouldin
    inertias = []
    silhouette_scores = []
    dbi_scores = []  # Davies–Bouldin Index
    cluster_range = range(2, 30)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df, labels))
        dbi_scores.append(davies_bouldin_score(df, labels))

    # Plot elbow method (inertia)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # Plot elbow method (inertia)
    axes[0].plot(cluster_range, inertias, marker='o')
    axes[0].set_title('Elbow Method - Inertia vs. Number of Clusters')
    axes[0].set_xlabel('Number of Clusters')
    axes[0].set_ylabel('Inertia')
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: f'{int(y)}'))
    axes[0].grid(True)

    # Plot silhouette score
    axes[1].plot(cluster_range, silhouette_scores, marker='o')
    axes[1].set_title('Silhouette Score vs. Number of Clusters (higher is better)')
    axes[1].set_xlabel('Number of Clusters')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: f'{int(y)}'))
    axes[1].grid(True)

    # Plot Davies–Bouldin Index
    axes[2].plot(cluster_range, dbi_scores, marker='o')
    axes[2].set_title('Davies–Bouldin Index vs. Number of Clusters (lower is better)')
    axes[2].set_xlabel('Number of Clusters')
    axes[2].set_ylabel('Davies–Bouldin Index')
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: f'{int(y)}'))
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()



complete_years = pd.read_csv(PATH_PROJECT / r'output\data\discharge_tables\complete_years.csv', index_col=0, dtype=str)

# for year in [2007, 2008]:
for year in complete_years.index:
    print(year)
    df = pd.read_csv(PATH_PROJECT / r'output\data\discharge_tables\discharge_table_2001_2022.csv', index_col=0, parse_dates=True)

    cids = complete_years.loc[year]
    cids = cids[~cids.isna()].values
    year_df = df.loc[str(year)][cids]

    year_df_scaled = ((year_df - year_df.mean()) / year_df.std(ddof=0)).copy()
    year_df_scaled = year_df_scaled.dropna(axis=1)
    elbow_silhouette_davies(year_df_scaled.T.copy())

    # input k
    k = int(input("Enter k: "))
    # k = 20
    # Apply KMeans clustering with 4 clusters (adjust if you prefer the k suggested above)
    kmeans_final = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_final = kmeans_final.fit_predict(year_df_scaled.T)

    # Reindex clusters by size: largest -> 0, second largest -> 1, ...
    labels_series = pd.Series(labels_final, index=year_df_scaled.T.index)
    order = labels_series.value_counts().index
    relabel_map = {old: new for new, old in enumerate(order)}
    labels_reindexed = labels_series.map(relabel_map)

    # Assign cluster labels to each catchment
    clustered_data = pd.DataFrame(index=year_df_scaled.T.index)
    clustered_data['Cluster'] = labels_reindexed
    clustered_data.sort_values(by='Cluster', inplace=True)

    file = out_path / f'KMeans_clustering_year_{str(year)}.csv'
    clustered_data.to_csv(file, index=True)

    cluster_file = str(file)
    station_mapping = pd.read_csv(os.getenv('STATION_MAPPING'), dtype=str)
    shape_file = Path(os.getenv('SHAPE_FILE'))
    all_catchments_gdf = geopandas.read_file(shape_file)
    cluster_mapping = pd.read_csv(cluster_file, dtype=str)
    cluster_mapping.columns = ['id', 'cluster']
    new_column = []
    for catchment in cluster_mapping['id'].values:
        new_column.append(np.int64(station_mapping[station_mapping["obsstednr"] == catchment]["Id15_v30"].values[0]))

    cluster_mapping['Id15_v30'] = new_column
    cluster_mapping.drop(columns=['id'], inplace=True)

    # filter out gauged catchments
    idxs = all_catchments_gdf['Id15_v30'].isin([np.int64(x) for x in cluster_mapping['Id15_v30'].values])
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
        legend=False,  # turn off auto legend
        cmap=cmap
    )

    ax.set_title(f"{year} KMeans clustering")

    # Add Denmark boundary
    denmark_boundary = all_catchments_gdf.boundary
    denmark_boundary.plot(ax=ax, color='black', linewidth=0.02)

    # Axis settings
    ax.set_xlim(left=440000, right=730000)
    ax.set_ylim(top=6.4e6, bottom=6.05e6)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # --------------------------------------------------------
    #  CUSTOM LEGEND (3 columns, small size, upper-right)
    # --------------------------------------------------------
    import matplotlib as mpl
    import matplotlib.patches as mpatches

    # Treat cluster as categorical so we have a stable order
    cluster_cat = merged_gdf["cluster"].astype("category")
    clusters_ordered = list(cluster_cat.cat.categories)  # ordered categories

    # New style colormap access (no deprecation warning)
    cmap_obj = mpl.colormaps[cmap]  # or mpl.colormaps.get_cmap(cmap)

    n_clusters = len(clusters_ordered)

    handles = []
    for i, c in enumerate(clusters_ordered):
        # Map category index to [0, 1] range for the colormap
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
        ncol=2,  # ⭐ three columns
        fontsize=6,  # small text
        title_fontsize=7,
        frameon=True,
        borderpad=0.3,
        handlelength=1.0
    )

    plt.tight_layout()
    plt.savefig(
        str(Path(cluster_file).parent / Path(cluster_file).stem) + '.png',
        dpi=300, bbox_inches='tight'
    )
    plt.show()




    # --- Load and preprocess data ---
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
    plt.axhline(y=0.9, color='r', linestyle='--', label=f'90% Variance. Components: {sum(~(cumulative_variance > 0.90))}')
    plt.axhline(y=0.95, color='g', linestyle='--', label=f'95% Variance. Components: {sum(~(cumulative_variance > 0.95))}')
    plt.axhline(y=0.98, color='y', linestyle='--', label=f'98% Variance. Components: {sum(~(cumulative_variance > 0.98))}')
    plt.legend()
    plt.tight_layout()
    plt.show()


    #  input number of components
    n_components = int(input("Enter number of components: "))
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df_scaled.T)
    pca_result = pd.DataFrame(pca_result, columns=[f'PCA{x}' for x in range(1, n_components +1)], index=df_scaled.T.index)


    # Transpose the data to have catchments as rows and days as features
    data = pca_result.copy()

    # Normalize each catchment's time series
    # scaler = StandardScaler()
    # data_scaled = scaler.fit_transform(data)
    data_scaled = (data - data.mean()) / data.std(ddof=0)
    data_scaled = data_scaled.dropna(axis=1)
    elbow_silhouette_davies(data_scaled.copy())

    k = int(input("Enter k: "))
    # k = 20
    # Apply KMeans clustering with 4 clusters (adjust if you prefer the k suggested above)
    kmeans_final = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_final = kmeans_final.fit_predict(data_scaled)

    # Reindex clusters by size: largest -> 0, second largest -> 1, ...
    labels_series = pd.Series(labels_final, index=data_scaled.index)
    order = labels_series.value_counts().index
    relabel_map = {old: new for new, old in enumerate(order)}
    labels_reindexed = labels_series.map(relabel_map)

    # Assign cluster labels to each catchment
    clustered_data = pd.DataFrame(index=data_scaled.index)
    clustered_data['Cluster'] = labels_reindexed
    clustered_data.sort_values(by='Cluster', inplace=True)

    file = out_path / f'KMeans_clustering_year_pca_{str(year)}.csv'
    clustered_data.to_csv(file, index=True)

    from pathlib import Path

    cluster_file = str(file)
    station_mapping = pd.read_csv(os.getenv('STATION_MAPPING'), dtype=str)
    shape_file = Path(os.getenv('SHAPE_FILE'))
    all_catchments_gdf = geopandas.read_file(shape_file)
    cluster_mapping = pd.read_csv(cluster_file, dtype=str)
    cluster_mapping.columns = ['id', 'cluster']
    new_column = []
    for catchment in cluster_mapping['id'].values:
        new_column.append(np.int64(station_mapping[station_mapping["obsstednr"] == catchment]["Id15_v30"].values[0]))

    cluster_mapping['Id15_v30'] = new_column
    cluster_mapping.drop(columns=['id'], inplace=True)

    # filter out gauged catchments
    idxs = all_catchments_gdf['Id15_v30'].isin([np.int64(x) for x in cluster_mapping['Id15_v30'].values])
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
        legend=False,  # turn off auto legend
        cmap=cmap
    )

    ax.set_title(f"{year} KMeans clustering PCA")

    # Add Denmark boundary
    denmark_boundary = all_catchments_gdf.boundary
    denmark_boundary.plot(ax=ax, color='black', linewidth=0.02)

    # Axis settings
    ax.set_xlim(left=440000, right=730000)
    ax.set_ylim(top=6.4e6, bottom=6.05e6)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # --------------------------------------------------------
    #  CUSTOM LEGEND (3 columns, small size, upper-right)
    # --------------------------------------------------------
    import matplotlib as mpl
    import matplotlib.patches as mpatches

    # Treat cluster as categorical so we have a stable order
    cluster_cat = merged_gdf["cluster"].astype("category")
    clusters_ordered = list(cluster_cat.cat.categories)  # ordered categories

    # New style colormap access (no deprecation warning)
    cmap_obj = mpl.colormaps[cmap]  # or mpl.colormaps.get_cmap(cmap)

    n_clusters = len(clusters_ordered)

    handles = []
    for i, c in enumerate(clusters_ordered):
        # Map category index to [0, 1] range for the colormap
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
        ncol=2,  # ⭐ three columns
        fontsize=6,  # small text
        title_fontsize=7,
        frameon=True,
        borderpad=0.3,
        handlelength=1.0
    )

    plt.tight_layout()
    plt.savefig(
        str(Path(cluster_file).parent / Path(cluster_file).stem) + '.png',
        dpi=300, bbox_inches='tight'
    )
    plt.show()
