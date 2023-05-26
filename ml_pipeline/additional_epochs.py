import os
import datetime
from tensorflow import keras
from keras.utils import image_dataset_from_directory
from keras.callbacks import TensorBoard
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from app.utils import get_first_file_path


# config
epochs = 30
root_path = 'models'

# get the most recent model
path, filename = get_first_file_path(root_path)
print("first_file: ", path)
model = keras.models.load_model(path)
print(model.summary())

# define dataset
data_directory = f"data/all-main-gcp-augmented2x-{datetime.date.today().strftime('%Y%m%d')}"
dataset, val_dataset = image_dataset_from_directory(
    data_directory,
    batch_size=32,
    image_size=(180, 320),
    subset="both",
    validation_split=0.2,
    seed=123,
    label_mode='categorical'
)

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

# define the callbacks
tensorboard_callback = TensorBoard(log_dir="./logs")

model.fit(
    dataset,
    epochs=epochs,
    validation_data=val_dataset,
    class_weight=class_weights_dict,
    callbacks=[tensorboard_callback]
)

model.save(os.path.join(root_path, f"add{epochs}-{filename}"))
