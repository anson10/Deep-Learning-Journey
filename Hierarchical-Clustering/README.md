# Hierarchical Clustering: A Comprehensive Explanation

Hierarchical clustering is a method of grouping data into a hierarchy of clusters based on their similarity. Unlike methods such as K-means, hierarchical clustering does not require you to predefine the number of clusters. Instead, it builds a **tree-like structure** called a dendrogram, which represents the hierarchy of clusters.

---

## Types of Hierarchical Clustering

### 1. **Agglomerative Hierarchical Clustering**
- **Bottom-up approach**: Each data point starts as its own cluster.
- Clusters are iteratively merged based on their similarity until all points belong to a single cluster.
  
### 2. **Divisive Hierarchical Clustering**
- **Top-down approach**: All points start in a single cluster.
- Clusters are recursively split until each data point forms its own cluster.

---

## Steps in Agglomerative Hierarchical Clustering

### 1. **Compute the Distance Matrix**
The first step is to compute the pairwise distances between all data points. The most common metric is the **Euclidean distance**, defined as:

$$
d(A, B) = \sqrt{\sum_{i=1}^n (B_i - A_i)^2}
$$

Where:
- $A = (A_1, A_2, \dots, A_n)$ are the coordinates of point $A$.
- $B = (B_1, B_2, \dots, B_n)$ are the coordinates of point $B$.
- $d(A, B)$ is the Euclidean distance between $A$ and $B$.

Other metrics such as Manhattan distance, cosine similarity, or Minkowski distance can also be used, depending on the problem.

### 2. **Merge the Closest Clusters**
- Initially, each point is its own cluster.
- Find the two closest clusters using a distance metric and merge them into a single cluster.
  
### 3. **Update the Distance Matrix**
After merging two clusters, update the distance matrix to reflect the distances between the new cluster and the remaining clusters. This step depends on the **linkage criterion**.

---

## Linkage Criteria

The linkage criterion determines how the distance between two clusters is computed. Common linkage methods include:

### 1. **Single Linkage**
The distance between two clusters is the **minimum distance** between any two points in the clusters:

$$
d(C_1, C_2) = \min \{ d(a, b) \mid a \in C_1, b \in C_2 \}
$$

- Results in elongated clusters.
- Can suffer from the **chaining effect**.

### 2. **Complete Linkage**
The distance between two clusters is the **maximum distance** between any two points in the clusters:

$$
d(C_1, C_2) = \max \{ d(a, b) \mid a \in C_1, b \in C_2 \}
$$

- Results in compact, spherical clusters.

### 3. **Average Linkage**
The distance between two clusters is the **average distance** between all pairs of points in the clusters:

$$
d(C_1, C_2) = \frac{1}{|C_1| \cdot |C_2|} \sum_{a \in C_1} \sum_{b \in C_2} d(a, b)
$$

Where:
- $|C_1|$ and $|C_2|$ are the sizes of clusters $C_1$ and $C_2$.

### 4. **Centroid Linkage**
The distance between two clusters is the distance between their centroids (mean points):

$$
d(C_1, C_2) = d(\text{centroid}_{C_1}, \text{centroid}_{C_2})
$$

Where the centroid of a cluster $C$ is given by:

$$
\text{centroid}_C = \frac{1}{|C|} \sum_{a \in C} a
$$

### 5. **Ward's Linkage**
Ward's method minimizes the **variance** within clusters by choosing clusters to merge based on minimizing the increase in total within-cluster variance. The increase in variance is computed as:

$$
\Delta E = \sum_{i \in C_1 \cup C_2} \| x_i - \text{centroid}_{C_1 \cup C_2} \|^2 - \sum_{i \in C_1} \| x_i - \text{centroid}_{C_1} \|^2 - \sum_{i \in C_2} \| x_i - \text{centroid}_{C_2} \|^2
$$

---

## Dendrogram

The dendrogram is a **tree-like structure** that represents the hierarchical clustering process. It provides insight into the hierarchy of clusters and can be used to decide the optimal number of clusters by "cutting" it at a certain height.

- The **height** at which two clusters are merged represents the distance or dissimilarity between them.
- Cutting the dendrogram at a specific level divides the data into clusters.

---

## Advantages of Hierarchical Clustering

1. **No Need to Predefine Clusters**:
   - Unlike K-means, hierarchical clustering does not require the number of clusters to be specified beforehand.

2. **Hierarchy Representation**:
   - Provides a dendrogram that gives insight into the data's structure.

3. **Non-Spherical Clusters**:
   - Can capture complex relationships and non-spherical clusters, unlike K-means.

---

## Disadvantages of Hierarchical Clustering

1. **Computational Complexity**:
   - Hierarchical clustering is computationally expensive with a complexity of $O(n^3)$ for $n$ data points.

2. **Irreversible Merges**:
   - Once two clusters are merged, they cannot be split later. Poor early decisions can affect the final result.

3. **Sensitive to Noise**:
   - Outliers can significantly affect clustering results.

---

## Applications of Hierarchical Clustering

1. **Genomics**:
   - Grouping genes or proteins based on similarity in expression profiles.

2. **Market Segmentation**:
   - Segmenting customers based on purchasing behavior.

3. **Document Clustering**:
   - Grouping similar documents for topic modeling.

4. **Social Network Analysis**:
   - Identifying communities or clusters in social networks.

---

