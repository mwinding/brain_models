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
            data.append([ds.attrs['neurotransmitter'], ds.attrs['connector_id'], arr[middle_index-1]])
            data.append([ds.attrs['neurotransmitter'], ds.attrs['connector_id'], arr[middle_index+1]])

            if(group=='Glutamate'): # make classes a bit more balanced
                data.append([ds.attrs['neurotransmitter'], ds.attrs['connector_id'], arr[middle_index-2]])
                data.append([ds.attrs['neurotransmitter'], ds.attrs['connector_id'], arr[middle_index+2]])            
                data.append([ds.attrs['neurotransmitter'], ds.attrs['connector_id'], arr[middle_index-3]])
                data.append([ds.attrs['neurotransmitter'], ds.attrs['connector_id'], arr[middle_index+3]])

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

'''
# convert test data to 28x28 Tensors
X_test = test_data.to_numpy()
X_test = X_test.reshape(len(X_test[:, 0]), 28, 28)
#X_test = [tf.constant(image) for image in X_test]
'''

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

# split train and test data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)

# %%
# set up classifier

input_shape = X_train[0].shape

model = tf.keras.Sequential([

    # base CNN layers
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape = [input_shape[0], input_shape[1], 1]),
    layers.BatchNormalization(),
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Dropout(0.2),
    
    # head neural net layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.35),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.35),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.35),
    layers.Dense(len(y_train[0]), activation='softmax') # 3 required to classify as Ach, GABA, or Glut
])

# compile models
model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

# %%
# fit models
epochs = 50

# I found that one has to monitor early-stopping
# if it stops after just a few epochs, the model is not well generalized and performs poorly
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    mode='max'
)

history = model.fit(
    tf.stack(X_train), tf.stack(y_train),
    validation_data = [tf.stack(X_valid), tf.stack(y_valid)],
    epochs = epochs,
    callbacks = [early_stopping]
)

# %%
# plot loss and accuracy

# add model type here
model_type = '10CNNs-increasingFilters_32_64_128_256_512'

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
