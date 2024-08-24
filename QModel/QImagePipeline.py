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
from QDataPipeline import QDataPipeline


class QImagePipeline:
    def __init__(self, mode, filepath):
        self.__mode__ = mode
        self.__filepath__ = filepath
        self.__features__ = None
        self.__k__ = None
        self.__model__ = None
        self.__labels__ = None
        self.__predicted_label__ = None
        self.__content__ = self.load_content()

    def pipeline(self):
        self.__images__ = self.load_images()
        # Load pre-trained VGG16 model + higher level layers
        base_model = VGG16(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
        model = Model(inputs=base_model.input, outputs=base_model.output)

        # Preprocess images
        processed_images = self.preprocess_images(self.__images__)

        # Extract features
        self.__features__ = self.extract_features(processed_images, model)

        if self.__mode__ == "training":
            self.__k__ = self.find_optimal_clusters()
            self.__labels__, self.__model__ = self.predict_cluster(
                self.__features__, n_clusters=self.__k__
            )
        elif self.__mode__ == "testing":
            self.__predicted_label__ = self.__model__.predict(self.__features__)

    def predict_cluster(self, features, n_clusters=5):
        """Clusters reduced images into k-optimal clusters returning a model and labels from the model."""
        print(f"[INFO] Clustering with {n_clusters} clusters")
        model = KMeans(n_clusters=n_clusters)
        labels = model.fit_predict(features)
        return labels, model

    def fig2img(self, fig):
        """Convert a Matplotlib figure to a PIL Image and return it"""
        import io

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img

    def load_content(self):
        print(f"[INFO] Loading content from {self.__filepath__}")
        content = []
        directories = []
        if self.__mode__ == "training":
            for root, dirs, files in os.walk(self.__filepath__):
                for file in files:
                    if (
                        file.endswith(".csv")
                        and not file.endswith("_poi.csv")
                        and not file.endswith("_lower.csv")
                    ):
                        content.append(os.path.join(root, file))
                        directories.append(root)
            self.__content__ = content
            self.__directories__ = directories
        elif self.__mode__ == "testing":
            print(self.__filepath__)
            self.__content__ = self.__filepath__

    def load_images(self):
        """Loads images from content list and returns a list of images."""
        print("[INFO] Loading images")
        if self.__mode__ == "training":
            images = []
            for i, file in enumerate(tqdm(self.__content__, desc="<<Loading Images>>")):
                qdp = QDataPipeline(file)
                qdp.preprocess()
                data = qdp.__dataframe__["Dissipation"]
                fig, ax = plt.subplots()
                ax.plot(data)
                ax.axis("off")
                image = self.fig2img(fig)
                images.append(image)
                plt.close()
            self.___images__ = images
        elif self.__mode__ == "testing":
            qdp = QDataPipeline(self.__filepath__)
            qdp.preprocess()
            data = qdp.__dataframe__["Dissipation"]
            fig, ax = plt.subplots()
            ax.plot(data)
            ax.axis("off")
            image = self.fig2img(fig)
            plt.close()
            self.___images__ = image

    def preprocess_images(self, images, target_size=(224, 224)):
        print("[INFO] Preprocessing images")
        if self.__mode__ == "training":
            processed_images = []
            for img in tqdm(images, desc="<<Preprocessing Images>>"):
                img = img.resize(target_size)
                if img.mode != "RGB":  # Check if the image is not already in RGB mode
                    img = img.convert("RGB")  # Convert grayscale to RGB
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0
                processed_images.append(img_array)
            return np.vstack(processed_images)
        elif self.__mode__ == "testing":
            img = images
            img = img.resize(target_size)
            if img.mode != "RGB":  # Check if the image is not already in RGB mode
                img = img.convert("RGB")  # Convert grayscale to RGB
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            processed_images.append(img_array)
            return np.vstack(processed_images)

    def extract_features(self, images, model):
        print(f"[INFO] Extracting features using model {model}")
        features = model.predict(images, use_multiprocessing=True)
        return features.reshape((features.shape[0], -1))

    def find_optimal_clusters(self, features, min_k=2, max_k=10):
        """Finds and optimal number of clusters based on the maximal sillohoutte score of the kmeans
        model."""
        print(f"[INFO] Finding optimal_k with max_k={max_k}")
        silhouette_scores = []
        k_values = range(min_k, max_k + 1)

        for k in tqdm(k_values, desc="<<Optimal K>>"):
            kmeans = KMeans(n_clusters=k)
            labels = kmeans.fit_predict(features)
            silhouette_scores.append(silhouette_score(features, labels))

        plt.figure(figsize=(10, 5))
        plt.plot(k_values, silhouette_scores, "bx-")
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Analysis For Optimal K")
        plt.show()
        optimal_k = k_values[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters based on silhouette analysis: {optimal_k}")
        return optimal_k


if __name__ == "__main__":
    qip = QImagePipeline(
        mode="testing", filepath="content/dropbox_dump/00000/DD240125W1_A5_40P_3rd.csv"
    )
    qip.pipeline()
    print(qip.__predicted_label__)
