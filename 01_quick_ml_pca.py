from sklearn.datasets import load_boston
import pandas as pd

# ------------- LOAD DATA ------------- #
# arrays
boston = load_boston()
X = boston.data
y = boston.target

# pandas df
X_df = pd.DataFrame(X, columns=boston.feature_names)
X_df.head()


def principal_component_analysis_sklearn(data, n_components=1):
    """ PCA with SVD under the hood using sklearn library """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    data_std = StandardScaler().fit_transform(data)

    pca = PCA(n_components=n_components)
    pca.fit(data_std)

    variance_explained_ratio = pca.explained_variance_ratio_
    singular_values = pca.singular_values_
    principal_components = pca.components_

    transformed_data = pca.fit_transform(data_std)

    return {'variance_explained_ratio': variance_explained_ratio,
            'singular_values': singular_values,
            'principal_components': principal_components,
            'transformed_data': transformed_data}


def principal_component_analysis_numpy_svd(data):
    """ pca using numpy svd and making the transformation """
    import numpy as np

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

    return {'variance_explained_ratio': variance_explained_ratio,
            'singular_values': s,
            'principal_components': vh,
            'transformed_data': transformed_data}


pca_output_sk = principal_component_analysis_sklearn(X_df, n_components=len(X_df.columns))
pca_output_np = principal_component_analysis_numpy_svd(X)


# validate different functions
# print(pca_output_sk['variance_explained_ratio'])
# print(pca_output_np['variance_explained_ratio'])
#
# print(pca_output_sk['singular_values'])
# print(pca_output_np['singular_values'])
#
# print(pca_output_sk['principal_components'])
# print(pca_output_np['principal_components'])
#
# print(pd.DataFrame(pca_output_sk['transformed_data']).head())
# print(pd.DataFrame(pca_output_np['transformed_data']).head())


def plot_scree_cumsum_visual(pca_output):
    """ plot scree and cumulative sum from pca function output"""
    import matplotlib.pyplot as plt
    import numpy as np

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


plot_scree_cumsum_visual(pca_output_sk)
plot_scree_cumsum_visual(pca_output_np)
