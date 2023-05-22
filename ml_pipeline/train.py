import numpy as np
import tensorflow as tf
import keras_tuner
from tensorflow import keras
from keras.utils import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import TensorBoard

import ml_pipeline.customCNN as customCNN

# check GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# define dataset
data_directory = "data/augmented-dataset"
dataset, val_dataset = image_dataset_from_directory(
    data_directory,
    batch_size=32,
    image_size=(180, 320),
    subset="both",
    validation_split=0.2,
    seed=123,
    label_mode='categorical'
)

# get the test dataset of 4 images
test_dataset = image_dataset_from_directory(
    "data/predict-4-in-order",
    batch_size=4,
    image_size=(180, 320),
    label_mode='categorical'
)

# For demonstration, iterate over the batches yielded by the dataset. and \
# iterate over the training dataset and put the labels into a list
labels = []
for data, label in dataset:
    print(data.shape)  # (64, 200, 200, 3)
    # print(data.dtype)  # float32
    # print(label.shape)  # (64,)
    # print(labels.dtype)  # int32
    # print(data)
    # print("labels:", label)
    for single_label in label:
        # print(single_label)
        # print(single_label.dtype)
        labels.append(single_label.numpy())

# get the categorical labels
labels_categorical = np.argmax(labels, axis=1)
class_labels = np.unique(labels_categorical)

# compute the class weights
class_weights = compute_class_weight(
    'balanced',
    classes=class_labels,
    y=labels_categorical
)
class_weights_dict = dict(enumerate(class_weights))
print(class_weights_dict)

# for data, labels in dataset:
#   data = scaler(data)
#   print(data.dtype)  # float32
#   print(data)
#   print("labels:", labels)
#   print("shape:", data.shape)
#   print("min:", np.min(data))
#   print("max:", np.max(data))

# we expect our inputs to be 320x180 RGB images
inputs = keras.Input(shape=(180, 320, 3))

# build the model
cnn = customCNN.CustomCNN(inputs, 4)

# if desired, we can define some hyperparameters for search
cnn.hp_dict = {
    "conv": {
        "unit_min": 16,
        "unit_max": 64,
        "unit_step": 16,
        "kernal_min": 3,
        "kernal_max": 10,
        "kernal_step": 1,
    },
    "dense": {
        "min_power": 2,
        "max_power": 4,
        "step": 1,
    },
    "dropout": {
        "min": 0.2,
        "max": 0.6,
        "step": 0.1,
    },
}


# define the model function
def build_hp_model(hp):
    model = keras.Model(
        inputs=inputs,
        # plain model
        # outputs=cnn.build_model()
        # with hyperparameter tuning
        outputs=cnn.build_model(hp=hp)
    )
    model.compile(
        optimizer=hp.Choice('optimizer', ['adam', 'nadam', 'rmsprop']),
        loss='categorical_crossentropy',
        metrics=[keras.metrics.CategoricalAccuracy()]
    )
    return model


# initialize the tuner
tuner = keras_tuner.RandomSearch(
    build_hp_model,
    objective='val_categorical_accuracy',
    max_trials=10,
)

# define the callbacks
tensorboard_callback = TensorBoard(log_dir="./logs")

# search for the best model
tuner.search(
    dataset,
    epochs=50,
    validation_data=val_dataset,
    class_weight=class_weights_dict,
    callbacks=[tensorboard_callback]
)

# get the best model
model = tuner.get_best_models()[0]

# run the model on a single batch of data
# processed_data = model(list(dataset.take(1).as_numpy_iterator())[0][0])
# print(processed_data.shape)
# print(model.summary())

# compile the model
# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.Recall()]
# )

# plot the model
keras.utils.plot_model(model, show_shapes=True, to_file="model.png")

# train the model
# history = model.fit(dataset, epochs=50, validation_data=val_dataset, \
# class_weight=class_weights_dict, callbacks=[tensorboard_callback])

# evaluate the model
loss, acc = model.evaluate(val_dataset)  # returns loss and metrics
print("loss: %.2f" % loss)
print("acc: %.2f" % acc)
# print("rec: %.2f" % rec)

# predict for the test_dataset
predictions = model.predict(test_dataset)
# print(predictions.shape)
print(predictions)
