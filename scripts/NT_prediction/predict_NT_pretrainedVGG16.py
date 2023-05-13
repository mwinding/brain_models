# %%
#
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

import h5py
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def grayscale_to_rgb(grayscale_image):

    # Reshape the grayscale image to have an extra channel dimension
    grayscale_image = tf.expand_dims(grayscale_image, axis=-1)

    # resize to image size for VGG
    grayscale_image = tf.image.resize(grayscale_image, size=(224, 224))

    # Tile the grayscale image along the channel dimension to create an RGB image
    rgb_image = tf.tile(grayscale_image, [1, 1, 3])

    return(rgb_image)

# %%
# load training and test data
##############

#path = input("Enter path of training dataset (HDF5):")
path = 'data/NT_cubes_2022-10-21.hdf5'

# collect timepoint one for total time elapsed
t1 = datetime.now()

# pull data from HDF5
data = []
with h5py.File(path) as f:
    for group in ['Glutamate', 'Acetylcholine', 'GABA']:
        print(f'loading datasets from group: {group}...')
        for key in f[group].keys():
            ds = f[f'{group}/{key}']
            arr = ds[:]
            middle_index = len(arr)//2 # if odd indices and if not, close enough

            # include middle z-slice and +1 / -1 z-slices as separate 2D images for training
            data.append([ds.attrs['neurotransmitter'], ds.attrs['connector_id'], arr[middle_index]])

            if(group=='Glutamate'): # make classes a bit more balanced
                data.append([ds.attrs['neurotransmitter'], ds.attrs['connector_id'], arr[middle_index-1]])
                data.append([ds.attrs['neurotransmitter'], ds.attrs['connector_id'], arr[middle_index+1]])            


train_data = pd.DataFrame(data, columns=['label', 'connector_id', 'array'])

# collect timepoint two for total time elapsed
t2 = datetime.now()

elapsed = t2-t1
elapsed = f'{elapsed.total_seconds()//60} min, {(elapsed.total_seconds()%60)//1} sec'


# %%
#

#test_data = pd.read_csv('data/handwritten-digits_MNIST/test.csv')

# shuffle values in train_data
train_data = train_data.sample(frac=1)
train_data = train_data.reset_index(drop=True)

# convert to 145x145 Tensors
X = train_data.drop(['label', 'connector_id'], axis=1).array.values
X = [tf.constant(image) for image in X]

y = train_data.loc[:, 'label']

# plot a few examples

nrows = 5
ncols = 5
plt.figure(figsize=(nrows*2,ncols*2))
for i in range(nrows*ncols):
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(255-X[i], cmap='Greys')
    plt.axis('off')
    plt.text(145/2, -10, str(y[i]), horizontalalignment='center', verticalalignment='center') # plot label above image
plt.savefig('plots/example_training_NT.png', format='png', bbox_inches='tight')
plt.show()

y = pd.get_dummies(y).to_numpy() # one-hot encoded to be compatible with model

# convert to rgb (duplicated 1-channel 3 times) to make compatible with pretrained VGG
X_rgb = [grayscale_to_rgb(x) for x in X]

# split train and test data
X_train_rgb, X_valid_rgb, y_train, y_valid = train_test_split(X_rgb, y, test_size=0.25, random_state=42)

# %%
# pre-trained model

from tensorflow.keras.applications import VGG16

# pretrained vgg model
def build_pretrained_vgg_model(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers of the pre-trained VGG model
    for layer in base_model.layers:
        layer.trainable = False
        
    model = tf.keras.Sequential()
    model.add(base_model)

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(len(y_train[0]), activation='softmax')) # 3 required to classify as Ach, GABA, or Glut

    return model

# Build the pre-trained VGG model with the given input shape
input_shape = X_train_rgb[0].shape
model = build_pretrained_vgg_model(input_shape)

# Compile and print the model summary
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# %%
# fit model
epochs = 1000

# I found that one has to monitor early-stopping
# if it stops after just a few epochs, the model is not well generalized and performs poorly
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=100,
    restore_best_weights=True,
    mode='max'
)

history = model.fit(
    tf.stack(X_train_rgb), tf.stack(y_train),
    validation_data = [tf.stack(X_valid_rgb), tf.stack(y_valid)],
    epochs = epochs,
    callbacks = [early_stopping]
)

# %%
# plot loss and accuracy

# add model type here
model_type = 'pretrained_VGG16'

history_df = pd.DataFrame(history.history)

date = datetime.now().strftime('%Y-%m-%d')
history_df.loc[0:, ['loss', 'val_loss']].plot()
plt.savefig(f'plots/{date}_{model_type}_train-validation_loss.png', format='png')
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))

history_df.loc[0:, ['accuracy', 'val_accuracy']].plot()
plt.savefig(f'plots/{date}_{model_type}_train-validation_accuracy.png', format='png')
print("Minimum validation loss: {}".format(history_df['val_accuracy'].max()))

history_df.to_csv(f'plots/history-df_{date}_{model_type}.csv')

'''
# make predictions with model

predictions = model.predict(X_test)
predictions = pd.DataFrame(predictions, index = test_data.index).round()

labels = []
for i in predictions.index:
    max_val = max(predictions.loc[i, :])
    num = np.where(predictions.loc[i, :]==max_val)[0][0]
    labels.append(num)

predictions = pd.DataFrame(predictions.index+1, columns=['ImageId'])
predictions['Label'] = labels
'''
# %%
