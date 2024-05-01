import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.metrics import silhouette_score, silhouette_samples

class Clustering:
    def __init__(self, dataset):
        self.dataset = dataset
        self.best_num_clusters = None
        self.silhouette_score = None
        
    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def kmeans(self, dataset, n_clusters, max_iterations=100):
        # Initialize the centroid randomly
        centroids = dataset[np.random.choice(dataset.shape[0], n_clusters, replace=False)]
        labels = np.zeros(len(dataset))
        
        for _ in range(max_iterations):
            # Calculate the distance of each input data to each centroid
            for i, point in enumerate(dataset):
                distances = [self.euclidean_distance(point, centroid) for centroid in centroids]
                labels[i] = np.argmin(distances)
                
            # Set the new centroid value
            new_centroids = np.array([dataset[labels == i].mean(axis=0) for i in range(n_clusters)])
            
            # Check if the centroid has converged
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids

        return labels
    
    def kmedoids(self, dataset, n_clusters, max_iterations=100):
        num_samples, num_features = dataset.shape
        
        # Initialize the medoids randomly
        medoids = np.random.choice(range(num_samples), n_clusters, replace=False)
        labels = np.zeros(num_samples)
        
        # Calculate the distance matrix
        distances = np.linalg.norm(dataset[:, np.newaxis, :] - dataset[np.newaxis, :, :], axis=-1)
        
        # Calculate the distance of each input data to each medoids
        for _ in range(max_iterations):
            distances_to_medoids = distances[medoids, :]
            labels = np.argmin(distances_to_medoids, axis=0)
            
            # Determine the new medoids value
            new_medoids = np.copy(medoids)
            for i in range(n_clusters):
                cluster_indices = np.where(labels == i)[0]
                cluster_distances = distances[cluster_indices][:, cluster_indices]
                costs = np.sum(cluster_distances, axis=1)
                new_medoids[i] = cluster_indices[np.argmin(costs)]
            
            if np.array_equal(medoids, new_medoids):
                break
            
            medoids = new_medoids

        return labels
    
    def silhouette(self, dataset, n_clusters):
        silhouette_avg = silhouette_score(dataset, n_clusters)
        return silhouette_avg
    
    def count_clusters(self, dataset):
        # Count the number of occurrences of each cluster label
        cluster_counts = dataset['Cluster Labels'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster Label', 'Count']
        return cluster_counts
    
    def apply_clustering(self, cluster_type='K-Means', min_clusters=2, max_clusters=10, sample_size=1000, random_state=200):
        # Exclude non-numeric columns from the dataset
        numeric_dataset = self.dataset.select_dtypes(include=[np.number])
        numeric_dataset = numeric_dataset.dropna()  # Drop rows with missing values

        # Convert min_clusters and max_clusters to integers
        min_clusters = int(min_clusters)
        max_clusters = int(max_clusters)
        silhouette_scores = []

        # Initialize best silhouette score and number of clusters
        max_silhouette_score = -1
        best_num_clusters = -1

        for i in range(min_clusters, max_clusters + 1):
            if cluster_type == 'K-Means':
                labels = self.kmeans(numeric_dataset.values, n_clusters=i)
            elif cluster_type == 'K-Medoids':
                labels = self.kmedoids(numeric_dataset.values, n_clusters=i)
            else:
                raise ValueError("Invalid cluster_type. Use 'K-Means' or 'K-Medoids'.")

            # Update cluster labels to start from 1 and increase sequentially
            labels = labels + 1

            silhouette_avg = self.silhouette(numeric_dataset.values, labels)

            # Update best silhouette score and number of clusters
            if silhouette_avg > max_silhouette_score:
                max_silhouette_score = silhouette_avg
                best_num_clusters = i

            silhouette_scores.append(silhouette_avg)

        # Plot the silhouette scores
        plt.figure(figsize=(10, 5))
        plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs Number of Clusters')
        st.pyplot(plt)

        # Initialize layout for silhouette diagrams
        num_rows = (max_clusters - min_clusters + 1) // 2 + ((max_clusters - min_clusters + 1) % 2 > 0)
        num_cols = 3
        layout = st.columns(num_cols)

        # Calculate silhouette diagrams and display
        for i in range(min_clusters, max_clusters + 1):
            if cluster_type == 'K-Means':
                labels = self.kmeans(numeric_dataset.values, n_clusters=i)
            elif cluster_type == 'K-Medoids':
                labels = self.kmedoids(numeric_dataset.values, n_clusters=i)

            # Update cluster labels to start from 1 and increase sequentially
            labels = labels + 1 - labels.min()

            # Plot Silhouette diagram
            fig, ax1 = plt.subplots()
            fig.set_size_inches(10, 7)

            ax1.set_xlim([-0.1, 1])
            ax1.set_ylim([0, len(numeric_dataset) + (i + 1) * 10])

            sample_silhouette_values = silhouette_samples(numeric_dataset, labels)

            y_lower = 10
            for j in range(i):
                ith_cluster_silhouette_values = sample_silhouette_values[labels == (j + 1)]

                ith_cluster_silhouette_values.sort()

                size_cluster_j = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_j

                color = plt.cm.nipy_spectral(float(j) / i)
                ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                                edgecolor=color, alpha=0.7)

                ax1.text(-0.05, y_lower + 0.5 * size_cluster_j, str(j + 1))
                y_lower = y_upper + 10

            ax1.set_title(f"The silhouette plot for {i} clusters.", fontsize=26)
            ax1.set_xlabel("The silhouette coefficient values", fontsize=22)
            ax1.set_ylabel("Cluster label", fontsize=22)

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_scores[i - min_clusters], color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # Display in columns
            layout[(i - min_clusters) % num_cols].pyplot(fig)
            st.write(f"Silhouette Score for {i} clusters: {silhouette_scores[i - min_clusters]}")

            # Close the figure to avoid overlapping in the next iteration
            plt.close(fig)
            
            # Display best results
        st.success(f"Best Silhouette Score: {max_silhouette_score} for {best_num_clusters} clusters")

        self.best_num_clusters = best_num_clusters
        self.silhouette_score = max_silhouette_score

        # Perform clustering with the best number of clusters
        if cluster_type == 'K-Means':
            labels = self.kmeans(numeric_dataset.values, n_clusters=self.best_num_clusters)
        elif cluster_type == 'K-Medoids':
            labels = self.kmedoids(numeric_dataset.values, n_clusters=self.best_num_clusters)

        # Update cluster labels to start from 1 and increase sequentially
        labels = labels + 1 - labels.min()

        # Calculate distances to centroids or medoids
        if cluster_type == 'K-Means':
            centroids = np.array([numeric_dataset.values[labels == i].mean(axis=0) for i in range(1, self.best_num_clusters + 1)])
            distances = np.linalg.norm(numeric_dataset.values[:, np.newaxis, :] - centroids, axis=-1)
            distance_cols = [f"C{i}" for i in range(1, self.best_num_clusters + 1)]
        elif cluster_type == 'K-Medoids':
            medoids = np.array([np.where(labels == i)[0][0] for i in range(1, self.best_num_clusters + 1)])
            distances = np.linalg.norm(numeric_dataset.values[:, np.newaxis, :] - numeric_dataset.values[medoids], axis=-1)
            distance_cols = [f"C{i}" for i in range(1, self.best_num_clusters + 1)]

        # Create a new DataFrame with original data, assigned cluster labels, and distances to centroids or medoids
        result_dataset_clustering = pd.concat([numeric_dataset, pd.DataFrame({'Cluster Labels': labels})], axis=1)
        result_dataset_clustering[distance_cols] = pd.DataFrame(distances, index=numeric_dataset.index)
        
        # Reorder columns
        new_columns = [col for col in result_dataset_clustering.columns if col not in ['Cluster Labels'] + distance_cols] + distance_cols + ['Cluster Labels']
        result_dataset_clustering = result_dataset_clustering.reindex(columns=new_columns)

        # Display the clustered data in a table
        st.subheader(f'{cluster_type} Clustering (Best Number of Clusters: {self.best_num_clusters})')
        st.dataframe(result_dataset_clustering)
        
        # Count cluster occurrences
        cluster_counts = self.count_clusters(result_dataset_clustering)
        cluster_counts_sorted = cluster_counts.sort_values(by='Cluster Label')  # Sorting by Cluster Label
        st.dataframe(cluster_counts_sorted)
        
        # Additional: Calculate cost and display if cluster_type is K-Medoids
        if cluster_type == 'K-Medoids':
            cluster_indices = [np.where(labels == i)[0] for i in range(1, self.best_num_clusters + 1)]
            cluster_distances = [np.linalg.norm(numeric_dataset.values[indices][:, np.newaxis, :] - numeric_dataset.values[medoids[i]], axis=-1).sum() for i, indices in enumerate(cluster_indices)]
            total_cost = sum(cluster_distances)
            st.write(f"Total cost: {total_cost}")
            
        scatter_fig_3d = px.scatter_3d(result_dataset_clustering, x=result_dataset_clustering.columns[0],
                                        y=result_dataset_clustering.columns[1], z=result_dataset_clustering.columns[2],
                                        color='Cluster Labels', size_max=18, opacity=0.7)
        scatter_fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(scatter_fig_3d)