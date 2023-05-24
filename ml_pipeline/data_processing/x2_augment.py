import ml_pipeline.augmentation as aug
import numpy as np
from keras.utils import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight


data_directory = "data/main-dataset"
output_directory = "data/augmented"

batch_size = 32


def augment_2x(
    source_folder: str,
    output_folder: str,
    batch_size: int = 32,
):
    class_strings = {
        0: "class_bed",
        1: "class_missing",
        2: "class_rug",
        3: "class_somewhere",
    }

    # get the dataset
    dataset, val_dataset = image_dataset_from_directory(
        source_folder,
        batch_size=32,
        image_size=(180, 320),
        subset="both",
        validation_split=0.2,
        seed=123,
        label_mode='categorical'
    )

    # compute the class weights
    labels = []
    for data, label in dataset:
        # print(data.shape)  # (64, 200, 200, 3)
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

    # get the number of files of each class as a dict
    number_each_dict = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    }
    for label in labels_categorical:
        number_each_dict[label] += 1
    print("number_each_dict: ", number_each_dict)

    # compute the class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=class_labels,
        y=labels_categorical
    )
    class_weights_dict = dict(enumerate(class_weights))
    print("class_weights_dict: ", class_weights_dict)

    # find the class with the most number of images
    min_class = min(class_weights_dict, key=class_weights_dict.get)
    # print("min_class: ", min_class)

    # find the factors by which to multiply the other classes
    factors = {}
    for key, value in class_weights_dict.items():
        if value != class_weights_dict[min_class]:
            factors[key] = value / class_weights_dict[min_class]
    print("factors: ", factors)

    # find the number required for each class
    number_required_dict = {}
    for key, value in factors.items():
        number_required_dict[key] = int(number_each_dict[key] * value) \
            - number_each_dict[key]
    print("number_required_dict: ", number_required_dict)

    # find the number of batches required for each class
    number_batches_dict = {}
    for key, value in number_required_dict.items():
        number_batches_dict[key] = int(value / batch_size)
    print("number_batches_dict: ", number_batches_dict)

    # add 20 batches to each class to roughly 2x the augmentation
    for key, value in number_batches_dict.items():
        number_batches_dict[key] = value + 20
    number_batches_dict[1] = 20
    print("number_batches_dict: ", number_batches_dict)

    # run the augmentation with saves to the right dir
    for key, value in number_batches_dict.items():
        augmentation = aug.Augmentation(
            data_directory=data_directory,
            output_dir=f"{output_folder}/{class_strings[key]}"
        )
        augmentation.batch_size = batch_size
        augmentation.set_generator(
            save=True,
            classes=[class_strings[key]],
        )
        for i in range(value):
            print(f"augmenting for {class_strings[key]}: batch {i+1}")
            augmentation.generator.next()

augment_2x(
    source_folder=data_directory,
    output_folder=output_directory,
    batch_size=batch_size,
)
