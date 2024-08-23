import joblib
from PIL import Image
from tqdm import tqdm
import os
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import shutil
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from keras.applications import VGG16
from keras_preprocessing.image import img_to_array
from keras.models import Model
from bundle_files import bundle_csv_files
from split_directories import split_data


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
    directories = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if (
                file.endswith(".csv")
                and not file.endswith("_poi.csv")
                and not file.endswith("_lower.csv")
            ):
                content.append(os.path.join(root, file))
                directories.append(root)
    return content, directories


def load_images(content, size):
    print("[INFO] Loading images")
    images = []
    for i, file in enumerate(tqdm(content, desc="<<Loading Images>>")):
        if i >= size:
            print("[INFO] Breaking early")
            break
        df = pd.read_csv(file)
        dissipation = df["Dissipation"]
        # Render the plot to a NumPy array
        fig, ax = plt.subplots()
        ax.plot(dissipation)
        ax.axis("off")
        image = fig2img(fig)
        images.append(image)
        plt.close()
    return images


def create_and_copy_directories(output_dir, directories, labels):
    # Create a dictionary to track the directories for each label
    label_dirs = {}

    for i, label in enumerate(labels):
        label_dir = os.path.join(output_dir, f"label_{label}")
        if label not in label_dirs:
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            label_dirs[label] = label_dir

        # Copy the contents from the directories to the new label directory
        if i < len(directories):
            src_dir = directories[i]
            for item in os.listdir(src_dir):
                src_path = os.path.join(src_dir, item)
                dst_path = os.path.join(label_dirs[label], item)
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, dst_path)


# Preprocess images
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


# Feature extraction using pre-trained CNN
def extract_features(images, model):
    print(f"[INFO] Extracting features using model {model}")
    features = model.predict(images, use_multiprocessing=True)
    return features.reshape((features.shape[0], -1))


# Dimensionality reduction
def reduce_dimensionality(features, n_components=50):
    print(f"[INFO] Dimensionality reduction with {n_components} components")
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features


def pipeline(images):
    # Load pre-trained VGG16 model + higher level layers
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=base_model.input, outputs=base_model.output)

    # Preprocess images
    processed_images = preprocess_images(images)

    # Extract features
    features = extract_features(processed_images, model)

    # Reduce dimensionality
    # reduced_features = reduce_dimensionality(features, n_components=50)
    labels = kmeans_model.predict(features)
    return labels


if __name__ == "__main__":
    kmeans_model = joblib.load("QModel/SavedModels/cluster.joblib")
    content, directories = load_content("content/training_data/train_clusters")
    images = load_images(content, len(content))
    labels = pipeline(images)
    output_dir = "content"
    create_and_copy_directories(output_dir, directories, labels)
    src = output_dir + f"/label_{0}"
    bundle_csv_files(src)
    split_data(src, src + "/train", src + "/test", 0.7)
    src = output_dir + f"/label_{1}"
    bundle_csv_files(src)
    split_data(src, src + "/train", src + "/test", 0.7)
    src = output_dir + f"/label_{2}"
    bundle_csv_files(src)
    split_data(src, src + "/train", src + "/test", 0.7)
