import numpy as np

class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminant = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)


        #S_W, S_B
        mean_overall = np.mean(X, axis=0)
        S_W = np.zeroes((n_features, n_features)) #Size nxn matrix
        S_B = np.zeroes((n_features, n_features)) #Size nxn matrix

        for c in class_labels:
            X_c = X[y==c]
            mean_c = np.mean(X_c, axis=0)
            #This order of transposing to ensure a nxn since the last dimension of the first matrix must match the last dimension of the second matrix
            #(4, n_c)*(n_c, 4) = (4,4)
            S_W += (X_c - mean_c).T.dot((X_c - mean_c)) 

            n_c = X_c.shape[0]
            #(4,1)*(4,1)T = (4,4)
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += n_c * (mean_diff).dot(mean_diff.T)
        
        A = np.linalg.inv(S_W).dot(S_B)

        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.linear_discriminant = eigenvectors[0:self.n_components]


    def transform(self, X):
        # project data
        return np.dot(X, self.linear_discriminant.T)
