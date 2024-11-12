# Wine Dataset Clustering

This project applies clustering techniques to the Wine dataset, exploring KMeans, Agglomerative Clustering, and PCA for dimensionality reduction.

## Steps

1. **Download Data**: The dataset is downloaded from Kaggle using `kagglehub`.
2. **Preprocessing**: The data is scaled using `StandardScaler`.
3. **Elbow Method**: KMeans is used to determine the optimal number of clusters by plotting the WCSS.
4. **Dimensionality Reduction**: PCA is applied to reduce the data to 2D for visualization.
5. **Agglomerative Clustering**: Segments the data into clusters.
6. **Visualization**: The clusters are visualized in a 2D scatter plot using PCA components.

## Libraries Used

- `numpy`
- `pandas`
- `matplotlib`
- `sklearn`
- `scipy`
- `kagglehub`

## Output

- Elbow Method plot for determining optimal clusters.
- PCA-based 2D scatter plot of wine data with color-coded clusters.

## Conclusion

This project demonstrates clustering and visualization techniques to segment wine data using KMeans and Agglomerative Clustering.
