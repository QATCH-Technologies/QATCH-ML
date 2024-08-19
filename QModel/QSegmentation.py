import tensorflow as tf
from keras import layers, Model
from keras_preprocessing.image import img_to_array, load_img
import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import sys

np.set_printoptions(threshold=sys.maxsize)
CLASS = ["background", "poi_1", "poi_2", "poi_3", "poi_4", "poi_5", "poi_6"]
COLOR_TO_CLASS = {
    (255, 255, 255): 0,  # Background (white)
    (255, 0, 0): 1,  # Red
    (0, 0, 255): 2,  # Blue
    (0, 255, 0): 3,  # Green
    (128, 0, 128): 4,  # Purple
    (255, 165, 0): 5,  # Orange
    (0, 255, 255): 6,  # Cyan
}
CLASS_TO_COLOR = {
    0: (255, 255, 255),  # Background (white)
    1: (255, 0, 0),  # Red
    2: (0, 0, 255),  # Blue
    3: (0, 255, 0),  # Green
    4: (128, 0, 128),  # Purple
    5: (255, 165, 0),  # Orange
    6: (0, 255, 255),  # Cyan
}


def create_custom_dataset(X, y, batch_size=32, is_train=True, upsampling_factor=2):
    def generator():
        if is_train:
            # Flatten masks to get class labels
            y_flat = np.argmax(y, axis=-1) if y.ndim == 4 else y
            # Compute class weights for upsampling
            class_counts = np.bincount(y_flat.flatten(), minlength=7)
            max_count = max(class_counts)
            indices_upsampled = []

            for i in range(7):
                class_indices = np.where(y_flat.flatten() == i)[0]
                if len(class_indices) < max_count:
                    # Upsample the minority class
                    upsampled_indices = np.random.choice(
                        class_indices, size=max_count, replace=True
                    )
                else:
                    upsampled_indices = class_indices
                indices_upsampled.append(upsampled_indices)

            indices_upsampled = np.concatenate(indices_upsampled)
            np.random.shuffle(indices_upsampled)
            X_upsampled = X[indices_upsampled]
            y_upsampled = y[indices_upsampled]
        else:
            X_upsampled, y_upsampled = X, y

        while True:
            for start in range(0, len(X_upsampled), batch_size):
                end = min(start + batch_size, len(X_upsampled))
                yield X_upsampled[start:end], y_upsampled[start:end]

    dataset = tf.data.Dataset.from_generator(
        generator,
        (tf.float32, tf.int32),
        (tf.TensorShape([None, *X.shape[1:]]), tf.TensorShape([None, *y.shape[1:]])),
    )
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def build_unet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    def conv_block(x, num_filters):
        x = layers.Conv2D(num_filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(num_filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def encoder_block(x, num_filters):
        conv = conv_block(x, num_filters)
        pool = layers.MaxPooling2D((2, 2))(conv)
        return conv, pool

    def decoder_block(x, concat_tensor, num_filters):
        x = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(
            x
        )
        x = layers.concatenate([x, concat_tensor])
        x = conv_block(x, num_filters)
        return x

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bottleneck
    b1 = conv_block(p4, 1024)

    # Decoder
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Output layer
    if num_classes == 1:
        outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(d4)
    else:
        outputs = layers.Conv2D(num_classes, (1, 1), activation="softmax")(d4)

    model = Model(inputs, outputs)
    return model


def create_dataset(X, y, batch_size=32, is_train=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def preprocess_pil_image(pil_image, image_size=(128, 128)):
    pil_image = pil_image.resize(image_size)
    image_array = np.array(pil_image)
    image_array = image_array / 255.0  # Normalize to [0, 1]
    return image_array


def preprocess_images(pil_images, target_size=(224, 224)):
    print("[INFO] Preprocessing images")
    processed_images = []
    for img in tqdm(pil_images, desc="<<Preprocessing Images>>"):
        img = img.resize(target_size)
        if img.mode != "RGB":  # Check if the image is not already in RGB mode
            img = img.convert("RGB")  # Convert grayscale to RGB
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        processed_images.append(img_array)
    return np.vstack(processed_images)


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def load_content(data_dir):
    print(f"[INFO] Loading content from {data_dir}")
    content = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if (
                file.endswith(".csv")
                and not file.endswith("_poi.csv")
                and not file.endswith("_lower.csv")
            ):
                data_path = os.path.join(root, file)
                poi_path = data_path.replace(".csv", "_poi.csv")
                content.append((data_path, poi_path))
    return content


def load_images(content, size=-1):
    print(f"[INFO] Loading {size} images")
    if size == -1:
        size = len(content)
    images = []
    for i, (data_file, _) in enumerate(tqdm(content, desc="<<Loading Images>>")):
        if i >= size:
            print("[INFO] Breaking early")
            break
        df = pd.read_csv(data_file)
        dissipation = df["Dissipation"]
        # Render the plot to a NumPy array
        fig, ax = plt.subplots()
        ax.plot(dissipation, color="black")
        ax.axis("off")
        image = fig2img(fig)
        images.append(image)
        plt.close()
    return images


def load_masks(content, size=-1):
    print(f"[INFO] Loading {size} masks")
    if size == -1:
        size = len(content)
    images = []
    for i, (data_file, poi_file) in enumerate(tqdm(content, desc="<<Loading Masks>>")):
        if i >= size:
            print("[INFO] Breaking early")
            break
        data_df = pd.read_csv(data_file)
        pois_x = pd.read_csv(poi_file, header=None).values
        dissipation = data_df["Dissipation"]
        pois_y = np.interp(pois_x, np.arange(len(dissipation)), dissipation)
        colors = ["red", "blue", "green", "purple", "orange", "cyan"]
        fig, ax = plt.subplots()
        # ax.plot(dissipation, color='black')
        for i, (x, y, color) in enumerate(zip(pois_x, pois_y, colors)):
            ax.scatter(x, y, s=2, color=color, zorder=5)
        ax.axis("off")
        image = fig2img(fig)
        images.append(image)
        plt.close()
    return images


def map_color_to_class(mask_image, color_to_class):
    mask_array = np.array(mask_image)
    if mask_array.shape[2] == 4:  # Check if mask has an alpha channel
        mask_array = mask_array[:, :, :3]  # Keep only RGB channels
    labeled_mask = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.int32)

    # Loop through the color to class mapping
    for color, class_id in color_to_class.items():
        # Create a boolean mask for each color
        color_mask = np.all(mask_array == np.array(color), axis=-1)
        # Assign class_id to the corresponding pixels
        labeled_mask[color_mask] = class_id
        return labeled_mask


def preprocess_mask(pil_mask, target_size=(224, 224), num_classes=7):
    pil_mask = pil_mask.resize(target_size)
    mask_array = np.array(pil_mask)

    # Ensure that the mask values are integer class labels
    if COLOR_TO_CLASS:
        mask_array = map_color_to_class(pil_mask, COLOR_TO_CLASS)

    # For multi-class segmentation, the mask should be in integer format
    return mask_array


def convert_labels_to_image(label_array, class_to_color):
    # Create an empty RGB image
    height, width = label_array.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Map each class label to its corresponding color
    for class_id, color in class_to_color.items():
        color_mask = label_array == class_id
        color_image[color_mask] = color

    return Image.fromarray(color_image)


def check_if_predictions_are_blank(predictions):
    # Check if all values in the predictions array are zeros
    if np.all(predictions == 0):
        raise ValueError(
            "All predictions are zeros. This might indicate an issue with the model or prediction process."
        )
    else:
        print("Predictions contain non-zero values.")


def display_predictions(predictions, class_to_color):
    for i, prediction in enumerate(predictions):
        # Convert each prediction to a color image
        color_image = convert_labels_to_image(prediction, class_to_color)

        # Display the image
        plt.figure(figsize=(8, 8))
        plt.title(f"Prediction {i + 1}")
        plt.imshow(color_image)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    # Assuming `images_list` and `masks_list` are lists of PIL images
    # Example:
    content = load_content("content/cluster_train")
    num_classes = 7
    images_list = load_images(content)
    masks_list = load_masks(content)

    input_shape = (224, 224, 3)
    images_array = preprocess_images(images_list)
    masks_array = np.array(
        [preprocess_mask(mask, num_classes=num_classes) for mask in masks_list]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        images_array, masks_array, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size=0.5, random_state=42
    )
    # train_dataset = create_dataset(X_train, y_train, is_train=True)
    # val_dataset = create_dataset(X_val, y_val, is_train=False)
    # test_dataset = create_dataset(X_test, y_test, is_train=False)
    train_dataset = create_custom_dataset(
        X_train, y_train, batch_size=32, is_train=True
    )
    val_dataset = create_custom_dataset(X_val, y_val, batch_size=32, is_train=False)
    test_dataset = create_custom_dataset(X_test, y_test, batch_size=32, is_train=False)

    # Adjust based on your task (e.g., 7 for multi-class)
    model = build_unet(input_shape, num_classes)

    # Choose appropriate loss function and metrics
    if num_classes == 1:
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",  # For binary segmentation
            metrics=["accuracy"],
        )
    else:
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",  # For multi-class segmentation
            metrics=["accuracy"],
        )

    history = model.fit(
        train_dataset, epochs=20, validation_data=val_dataset
    )  # val_dataset is optional

    # Evaluate the model
    model.evaluate(test_dataset)
    test_predictions = model.predict(X_test)
    test_predictions = np.argmax(
        test_predictions, axis=-1
    )  # Convert logits to class labels if needed
    check_if_predictions_are_blank(test_predictions)

    # Display the predictions
    display_predictions(test_predictions, CLASS_TO_COLOR)
