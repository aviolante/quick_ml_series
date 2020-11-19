"""

This script, within each function, gives a step by step approach to do PCA using sklearn
('principal_component_analysis_sklearn') or to do it manually with numpy svd
('principal_component_analysis_numpy_svd'). I've wrapped the steps in functions for ease
of use, but it could easily be split out or put into some pipeline if desired. I've also
included a function ('plot_scree_cumsum_visual') to produce 2-plots: scree and cumulative
sum to decide on the size of the lower-dimensional representation.

"""

# import libraries
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# load data as array and pandas df
# arrays
boston = load_boston()
X = boston.data
y = boston.target

# pandas df
X_df = pd.DataFrame(X, columns=boston.feature_names)
X_df.head()


# helper functions for pca (numpy and pandas) and plotting scree / cumsum
def principal_component_analysis_sklearn(data, n_components=1):
    """ PCA with SVD under the hood using sklearn library """
    data_std = StandardScaler().fit_transform(data)

    # fit pca on data using sklearn
    pca = PCA(n_components=n_components)
    pca.fit(data_std)

    # assign values from pca object
    variance_explained_ratio = pca.explained_variance_ratio_
    singular_values = pca.singular_values_
    principal_components = pca.components_

    # apply transformation
    transformed_data = pca.fit_transform(data_std)

    # return dictionary of items of interest
    return {'variance_explained_ratio': variance_explained_ratio,
            'singular_values': singular_values,
            'principal_components': principal_components,
            'transformed_data': transformed_data}


def principal_component_analysis_numpy_svd(data):
    """ pca using numpy svd and making the transformation """
    # standardize data
    data_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # perform svd to decompose matrix
    u, s, vh = np.linalg.svd(data_std, full_matrices=False, compute_uv=True)

    # get variance and variance explained
    explained_variance = (s ** 2) / (len(data_std) - 1)
    total_variance = explained_variance.sum()
    variance_explained_ratio = explained_variance / total_variance

    # apply transformation
    transformed_data = np.dot(u, np.diag(s))

    # return dictionary of items of interest
    return {'variance_explained_ratio': variance_explained_ratio,
            'singular_values': s,
            'principal_components': vh,
            'transformed_data': transformed_data}


def plot_scree_cumsum_visual(pca_output):
    """ plot scree and cumulative sum from pca function output"""
    pc_num = len(pca_output['singular_values'])
    pc_values = np.arange(pc_num) + 1

    plt.subplot(1, 2, 1)
    plt.plot(pc_values, pca_output['variance_explained_ratio'] * 100, 'ro-', linewidth=1, markersize=5)
    plt.xticks(np.arange(1, pc_num + 1, step=1))
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained (%)')

    plt.subplot(1, 2, 2)
    plt.plot(pc_values, np.cumsum(pca_output['variance_explained_ratio']) * 100, 'bo-', linewidth=1, markersize=5)
    plt.xticks(np.arange(1, pc_num + 1, step=1))
    plt.title('Cumulative Sum Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained (%)')

    plt.tight_layout()
    plt.show()


# perform pca / svd
pca_output_sk = principal_component_analysis_sklearn(X_df, n_components=len(X_df.columns))  # n-components = n-features
pca_output_np = principal_component_analysis_numpy_svd(X)

# validate and compare different functions output
print(pca_output_sk['variance_explained_ratio'])
print(pca_output_np['variance_explained_ratio'])

# [0.47129606 0.11025193 0.0955859  0.06596732 0.06421661 0.05056978
#  0.04118124 0.03046902 0.02130333 0.01694137 0.0143088  0.01302331
#  0.00488533]
#
# [0.47129606 0.11025193 0.0955859  0.06596732 0.06421661 0.05056978
#  0.04118124 0.03046902 0.02130333 0.01694137 0.0143088  0.01302331
#  0.00488533]


print(pca_output_sk['singular_values'])
print(pca_output_np['singular_values'])

# [55.6793095  26.93022859 25.07516773 20.83105866 20.55278239 18.23864114
#  16.45874174 14.15716218 11.83779223 10.55653065  9.70171478  9.25566343
#   5.66883461]
#
# [55.6793095  26.93022859 25.07516773 20.83105866 20.55278239 18.23864114
#  16.45874174 14.15716218 11.83779223 10.55653065  9.70171478  9.25566343
#   5.66883461]

print(pca_output_sk['principal_components'])
print(pca_output_np['principal_components'])

print(pd.DataFrame(pca_output_sk['transformed_data']).head())
print(pd.DataFrame(pca_output_np['transformed_data']).head())

#          0         1         2   ...        10        11        12
# 0 -2.098297  0.773113  0.342943  ... -0.033000  0.019440  0.365975
# 1 -1.457252  0.591985 -0.695199  ... -0.640810 -0.125797 -0.070719
# 2 -2.074598  0.599639  0.167122  ... -0.487557  0.133327 -0.014022
# 3 -2.611504 -0.006871 -0.100284  ... -0.360209  0.508678  0.007847
# 4 -2.458185  0.097712 -0.075348  ... -0.395150  0.497732  0.014274
# [5 rows x 13 columns]
#
#          0         1         2   ...        10        11        12
# 0 -2.098297  0.773113  0.342943  ... -0.033000 -0.019440  0.365975
# 1 -1.457252  0.591985 -0.695199  ... -0.640810  0.125797 -0.070719
# 2 -2.074598  0.599639  0.167122  ... -0.487557 -0.133327 -0.014022
# 3 -2.611504 -0.006871 -0.100284  ... -0.360209 -0.508678  0.007847
# 4 -2.458185  0.097712 -0.075348  ... -0.395150 -0.497732  0.014274
# [5 rows x 13 columns]

# plot scree and cumulative sum plots
plot_scree_cumsum_visual(pca_output_sk)
plot_scree_cumsum_visual(pca_output_np)