# **Principal Component Analysis (PCA)**

Principal Component Analysis (PCA) is a statistical method used for **dimensionality reduction** and **feature extraction**. It transforms a dataset with potentially correlated variables into a set of new, uncorrelated variables called **principal components**, which capture the maximum variance in the data.

---

## **Steps in PCA**

### 1. **Standardize the Dataset**
Given a dataset $X$ with $n$ features and $m$ samples, we first standardize it to have zero mean and unit variance for each feature:
$$
Z_{ij} = \frac{X_{ij} - \mu_j}{\sigma_j}
$$
Where:
- $X_{ij}$: Value of the $j$-th feature for the $i$-th sample.
- $\mu_j$: Mean of the $j$-th feature.
- $\sigma_j$: Standard deviation of the $j$-th feature.

### 2. **Compute the Covariance Matrix**
The covariance matrix captures the relationships between the features:
$$
C = \frac{1}{m-1} Z^T Z
$$
Where:
- $Z$ is the standardized data matrix ($m \times n$).
- $C$ is an $n \times n$ matrix where each element $C_{ij}$ represents the covariance between feature $i$ and feature $j$.

### 3. **Compute Eigenvalues and Eigenvectors**
Solve the eigenvalue equation for the covariance matrix:
$$
C \mathbf{v} = \lambda \mathbf{v}
$$
Where:
- $C$: Covariance matrix.
- $\mathbf{v}$: Eigenvector (principal component direction).
- $\lambda$: Eigenvalue (variance along the eigenvector).

The eigenvalues represent the amount of variance captured by each eigenvector.

### 4. **Sort Eigenvalues and Eigenvectors**
Order the eigenvalues $\lambda_1, \lambda_2, \dots, \lambda_n$ in descending order and their corresponding eigenvectors. Select the top $k$ eigenvectors corresponding to the largest eigenvalues to form the **principal components**.

### 5. **Project the Data**
Transform the original data $Z$ onto the new basis defined by the selected eigenvectors:
$$
Z_{\text{PCA}} = Z W_k
$$
Where:
- $W_k$: Matrix of the top $k$ eigenvectors (size $n \times k$).
- $Z_{\text{PCA}}$: Transformed data in the reduced $k$-dimensional space.

---

## **Mathematical Interpretation**

### **Variance Maximization**
PCA finds the directions (principal components) that maximize the variance of the data:
$$
\mathbf{v}_1 = \underset{\mathbf{v}}{\text{argmax}} \ \text{Var}(Z \mathbf{v}), \quad \|\mathbf{v}\| = 1
$$

The principal components are orthogonal to each other and capture the majority of the variance in descending order:
$$
\text{Var}(\mathbf{v}_1) \geq \text{Var}(\mathbf{v}_2) \geq \dots \geq \text{Var}(\mathbf{v}_n)
$$

### **Minimizing Reconstruction Error**
PCA also minimizes the reconstruction error when reducing the dimensionality:
$$
\min \|Z - Z_{\text{PCA}} W_k^T\|^2
$$

---

## **Example Visualization**
- The **first principal component (PC1)** is the direction of maximum variance in the data.
- The **second principal component (PC2)** is orthogonal to PC1 and captures the second highest variance.

For $k < n$, PCA reduces the dimensions while retaining most of the variability.

---

## **Applications of PCA**
1. **Dimensionality Reduction**: Reducing the number of features while preserving variance.
2. **Data Visualization**: Visualizing high-dimensional data in 2D or 3D.
3. **Noise Reduction**: Eliminating components with low variance.
4. **Feature Extraction**: Identifying significant patterns or features.

---

## **Key Properties**
1. The principal components are **uncorrelated**.
2. The total variance is preserved:
   $$ 
   \text{Total Variance} = \sum_{i=1}^n \lambda_i
   $$
3. PCA assumes that the directions of maximum variance are the most important.
