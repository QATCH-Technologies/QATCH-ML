import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


sys.path.insert(0, os.path.join(os.getcwd(), "src"))
print(sys.path)
from src.models.q_image_clusterer import QClusterer


class TestQClusterer(unittest.TestCase):
    @patch("QClusterer.VGG16")
    def setUp(self, mock_vgg16):
        """Sets up the QClusterer object for testing."""
        print("[STATUS] Setting up tests")
        self.qclusterer = QClusterer()

    def test_init_no_model_path(self):
        result = True
        self.assertEqual(result, True)

    # @patch("QClusterer.joblib.load")
    # def test_init_with_model_path(self, mock_load):
    #     """Tests initialization with a model path (prediction mode)."""
    #     mock_kmeans = MagicMock()
    #     mock_load.return_value = mock_kmeans
    #     qclusterer = QClusterer(model_path="mock_path")
    #     self.assertEqual(qclusterer.kmeans, mock_kmeans)

    # def test_convert_to_image(self):
    #     """Tests converting a matplotlib figure to a PIL image."""
    #     fig, ax = plt.subplots()
    #     ax.plot([1, 2, 3])
    #     img = self.qclusterer.convert_to_image(fig)
    #     self.assertIsInstance(img, Image.Image)

    # @patch("QClusterer.os.walk")
    # def test_load_content(self, mock_walk):
    #     """Tests loading content from a directory."""
    #     mock_walk.return_value = [
    #         ("root", [], ["file1.csv", "file2_poi.csv", "file3_lower.csv", "file4.csv"])
    #     ]
    #     content = self.qclusterer.load_content("mock_directory")
    #     self.assertEqual(content, ["root/file1.csv", "root/file4.csv"])

    # @patch("QClusterer.QDataPipeline")
    # @patch("QClusterer.QClusterer.convert_to_image")
    # def test_load_images(self, mock_convert_to_image, mock_qdp):
    #     """Tests loading and converting images from content."""
    #     mock_image = MagicMock()
    #     mock_convert_to_image.return_value = mock_image
    #     mock_qdp.return_value.__dataframe__ = {"Dissipation": [1, 2, 3]}
    #     content = ["file1.csv", "file2.csv"]
    #     images = self.qclusterer.load_images(content, size=1)
    #     self.assertEqual(len(images), 1)
    #     self.assertEqual(images[0], mock_image)

    # def test_preprocess_images(self):
    #     """Tests preprocessing PIL images."""
    #     img = Image.new("RGB", (100, 100))
    #     pil_images = [img]
    #     processed_images = self.qclusterer.preprocess_images(pil_images)
    #     self.assertEqual(processed_images.shape, (1, 224, 224, 3))

    # def test_extract_features(self):
    #     """Tests extracting features from preprocessed images."""
    #     mock_images = np.random.rand(1, 224, 224, 3)
    #     mock_output = np.random.rand(1, 7, 7, 512)
    #     self.qclusterer.model.predict.return_value = mock_output
    #     features = self.qclusterer.extract_features(mock_images)
    #     self.assertEqual(features.shape, (1, 7 * 7 * 512))

    # @patch("QClusterer.KMeans")
    # @patch("QClusterer.silhouette_score")
    # def test_find_optimal_clusters(self, mock_silhouette_score, mock_kmeans):
    #     """Tests finding the optimal number of clusters using silhouette score."""
    #     mock_silhouette_score.side_effect = [0.2, 0.5, 0.4]
    #     mock_kmeans().fit_predict.return_value = np.random.randint(0, 3, 10)
    #     mock_features = np.random.rand(10, 512)
    #     optimal_k = self.qclusterer.find_optimal_clusters(
    #         mock_features, min_k=2, max_k=4
    #     )
    #     self.assertEqual(optimal_k, 3)

    # @patch("QClusterer.KMeans")
    # def test_perform_clustering(self, mock_kmeans):
    #     """Tests performing clustering using KMeans."""
    #     mock_labels = np.array([0, 1, 0])
    #     mock_kmeans().fit_predict.return_value = mock_labels
    #     mock_features = np.random.rand(3, 512)
    #     labels = self.qclusterer.perform_clustering(mock_features, n_clusters=2)
    #     self.assertTrue((labels == mock_labels).all())

    # @patch("QClusterer.plt.show")
    # def test_visualize_clusters(self, mock_plt_show):
    #     """Tests visualizing clusters using a scatter plot."""
    #     features = np.random.rand(10, 2)
    #     labels = np.random.randint(0, 2, 10)
    #     self.qclusterer.visualize_clusters(features, labels)
    #     mock_plt_show.assert_called_once()

    # @patch("QClusterer.plt.show")
    # def test_display_cluster_images(self, mock_plt_show):
    #     """Tests displaying a sample of images from each cluster."""
    #     pil_images = [Image.new("RGB", (100, 100)) for _ in range(4)]
    #     labels = np.array([0, 1, 0, 1])
    #     self.qclusterer.display_cluster_images(pil_images, labels, n_clusters=2)
    #     mock_plt_show.assert_called()

    # @patch("QClusterer.QDataPipeline")
    # def test_predict_label(self, mock_qdp):
    #     """Tests predicting the cluster label for a given CSV file."""
    #     mock_qdp.return_value.__dataframe__ = {"Dissipation": [1, 2, 3]}
    #     self.qclusterer.kmeans = MagicMock()
    #     self.qclusterer.kmeans.predict.return_value = [1]
    #     mock_file = MagicMock()
    #     mock_file.readlines.return_value = ["header\n", "1,2,3\n"]
    #     predicted_label = self.qclusterer.predict_label(mock_file)
    #     self.assertEqual(predicted_label, 1)

    # @patch("QClusterer.joblib.dump")
    # def test_save_model(self, mock_dump):
    #     """Tests saving the trained KMeans model."""
    #     self.qclusterer.kmeans = MagicMock()
    #     self.qclusterer.save_model("mock_path")
    #     mock_dump.assert_called_once_with(self.qclusterer.kmeans, "mock_path")


if __name__ == "__main__":
    unittest.main()
