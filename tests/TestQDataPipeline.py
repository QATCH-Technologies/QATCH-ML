import unittest
import os
import pandas as pd
from unittest.mock import patch, mock_open
from io import StringIO
from q_data_pipeline import QDataPipeline


class TestQDataPipeline(unittest.TestCase):

    def setUp(self):
        # Mock CSV data for testing
        self.mock_csv_data = "col1,col2,col3\n1,2,3\n4,5,6\n7,8,9"
        self.mock_filepath = "mock_data.csv"

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="col1,col2,col3\n1,2,3\n4,5,6\n7,8,9",
    )
    @patch("pandas.read_csv")
    def test_init_valid_filepath(self, mock_read_csv, mock_file):
        # Mock the pandas read_csv function to return a DataFrame
        mock_read_csv.return_value = pd.read_csv(StringIO(self.mock_csv_data))

        # Initialize the QDataPipeline object with a valid filepath
        pipeline = QDataPipeline(data_filepath=self.mock_filepath)

        # Check if the DataFrame was read and stored correctly
        mock_read_csv.assert_called_with(self.mock_filepath)
        self.assertIsNotNone(pipeline.__dataframe__)
        self.assertEqual(pipeline.multi_class, False)

    def test_init_missing_filepath(self):
        # Test the ValueError when no filepath is provided
        with self.assertRaises(ValueError) as context:
            QDataPipeline()

        self.assertTrue(
            "[QDataPipeline.__init__] Filepath required" in str(context.exception)
        )

    @patch("pandas.read_csv")
    def test_init_multi_class_flag(self, mock_read_csv):
        # Mock the pandas read_csv function to return a DataFrame
        mock_read_csv.return_value = pd.read_csv(StringIO(self.mock_csv_data))

        # Initialize the pipeline with multi_class set to True
        pipeline = QDataPipeline(data_filepath=self.mock_filepath, multi_class=True)

        # Check that the multi_class flag is set correctly
        self.assertTrue(pipeline.multi_class)


if __name__ == "__main__":
    unittest.main()
