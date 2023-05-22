import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


# This class is used to augment images
class Augmentation:

    def __init__(self, data_directory: str, output_dir: str) -> None:
        self.data_directory = data_directory
        self.output_dir = output_dir
        self.batch_size = 4
        self.set_datagen()

    # Set the image data generator up with the required ranges
    def set_datagen(
        self,
        rotation_range: int = 20,
        width_shift_range: float = 0.1,
        height_shift_range: float = 0.1,
        zoom_range: float = 0.05,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        brightness_range: tuple = (0.5, 1.5),
    ) -> None:
        self.datagen = ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            brightness_range=brightness_range,
        )
        self.set_generator()

    # Set the generator up with the data directory
    def set_generator(self, save: bool = False, classes: list = None) -> None:
        self.generator = self.datagen.flow_from_directory(
            self.data_directory,
            target_size=(180, 320),
            batch_size=self.batch_size,
            class_mode='categorical',
            save_to_dir=self.output_dir if save else None,
            save_format="jpg" if save else None,
            classes=classes if classes is not None else None,
        )

    # Plot the augmented images
    def plot_batches(self, rows: int = 4) -> None:
        fig, axes = plt.subplots(rows, self.batch_size, figsize=(30, 30))
        for r in range(rows):
            # batch of images
            images, labels = self.generator.next()
            # print(labels)

            # get the number returned
            images_count = images.shape[0]
            # print("images_count:", images_count)

            for c in range(images_count):
                # get the image
                image = images[c].astype('uint8')
                # show the image
                axes[r, c].imshow(image)
                axes[r, c].title.set_text(convert_label(labels[c]))

        plt.savefig("augmented_images.png")


def convert_label(labels: list) -> str:
    if labels[0] == 1:
        return "bed"
    elif labels[1] == 1:
        return "missing"
    elif labels[2] == 1:
        return "rug"
    elif labels[3] == 1:
        return "somewhere"
