import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.manifold import TSNE
import seaborn as sns

# load data, UCI Data from https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
bank_df = pd.read_csv('bank/bank-full.csv', delimiter=';')
bank_df.head()

# ohe categorical features in data (blindly)
bank_df_dummy = pd.get_dummies(bank_df)

# build ae architecture
input_dim = bank_df_dummy.shape[1]
encoding_dim = 10

input_layer = Input(shape=(input_dim,), name="input_layer")
encoder = Dense(units=encoding_dim * 2, activation="relu", name="encoder")(input_layer)
bottleneck = Dense(units=encoding_dim, activation="relu", name="bottleneck")(encoder)
decoder = Dense(units=encoding_dim * 2, activation="relu", name="decoder")(bottleneck)
output_layer = Dense(units=input_dim, activation="linear", name="output_layer")(decoder)

# construct model
autoencoder = Model(inputs=input_layer, outputs=output_layer, name="autoencoder")

# compile model
autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')

# train model
history = autoencoder.fit(bank_df_dummy, bank_df_dummy,
                          epochs=50,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(bank_df_dummy, bank_df_dummy),
                          verbose=1)

# plot model architecture and save as png
tf.keras.utils.plot_model(autoencoder,
                          to_file="ae_model.png",
                          show_shapes=True,
                          show_layer_activations=True)

# get layer weights and use tf.matmul, matrix multiplication, to "score data" you could also use Model() and pass
# the weights to the layers
embedding_matrix = autoencoder.get_layer('encoder').get_weights()[0]
embed_encode_data = tf.matmul(bank_df_dummy, embedding_matrix)
embedding_matrix = autoencoder.get_layer('bottleneck').get_weights()[0]
embed_encode_data = tf.matmul(embed_encode_data, embedding_matrix)
embed_encode_data = embed_encode_data.numpy()
embed_encode_data_df = pd.DataFrame(embed_encode_data)

# use t-sne to visualize and validate your autoencoder embedding or dense representation
tsne_embedding = TSNE(n_components=2, learning_rate=75, perplexity=15, init='random').fit_transform(embed_encode_data_df)

bank_df['tsne_dim1'] = tsne_embedding[:, 0]
bank_df['tsne_dim2'] = tsne_embedding[:, 1]

# plot visual of t-sne embedding (its fun!)
sns.scatterplot(x="tsne_dim1", y="tsne_dim2", hue='y',
                palette=sns.color_palette("Set2", 2),
                data=bank_df).set(title="Bank Data T-SNE Projection")


