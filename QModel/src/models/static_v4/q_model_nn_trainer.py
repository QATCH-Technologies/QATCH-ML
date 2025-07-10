import os
import pickle
import random
import logging
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import keras_tuner as kt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE

from q_model_data_processor import QDataProcessor
# from the previous snippet
from q_model_nn import SequenceClassifierHyperModel
import tensorflow as tf
print("TF version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("CUDA visible to TF:", tf.sysconfig.get_build_info().get("cuda_version"))
print("cuDNN version:", tf.sysconfig.get_build_info().get("cudnn_version"))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

RANDOM_STATE = 42


class TFSequenceTrainer:
    def __init__(
        self,
        classes: List[str],
        seq_len: int,
        regen_data: bool = False,
        train_dir: str = "content/static/train",
        val_dir:   str = "content/static/valid",
        test_dir:  str = "content/static/test",
        cache_dir: str = "cache/tf_sequence"
    ):
        self.classes = classes
        self.num_classes = len(classes)
        self.seq_len = seq_len

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.scaler_path = os.path.join(self.cache_dir, "scaler.pkl")
        self.scaler: Optional[Pipeline] = None

        self.regen_data = regen_data

        # placeholders
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        # load raw file lists via QDataProcessor
        # self.train_content = QDataProcessor.load_balanced_content(
        #     train_dir, num_datasets=np.inf, opt=True)[0]
        # self.val_content = QDataProcessor.load_balanced_content(
        #     val_dir,   num_datasets=np.inf, opt=True)[0]
        # self.test_content = QDataProcessor.load_balanced_content(
        #     test_dir,  num_datasets=np.inf, opt=True)[0]

        self.train_content = QDataProcessor.load_content(
            train_dir, num_datasets=30)
        self.val_content = QDataProcessor.load_content(
            val_dir,   num_datasets=20)
        self.test_content = QDataProcessor.load_content(
            test_dir,  num_datasets=20)

    def _get_flat_xy(self, content) -> Tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        for data_file, poi_file in tqdm(content, desc="<Flattening X, y vector>"):
            df = QDataProcessor.process_data(data_file)
            poi_df = QDataProcessor.process_poi(poi_file, length_df=len(df))
            df["POI"] = poi_df

            # FEATURES
            X_list.append(df.drop(columns="POI").values)

            # LABELS: force to 1-D
            # If poi_df is a DataFrame or ndarray of shape (n,1),
            # squeeze it down to (n,)
            y_arr = poi_df.values.squeeze()
            y_list.append(y_arr)

        X = np.vstack(X_list)        # shape: (total_samples, num_features)
        y = np.hstack(y_list)        # shape: (total_samples,)
        return X, y

    def _balance_and_scale(self, X: np.ndarray, y: np.ndarray, split_name: str):
        # 1) SMOTE
        sm = SMOTE(random_state=RANDOM_STATE)
        Xb, yb = sm.fit_resample(X, y)
        logging.info(f"{split_name} after SMOTE: {np.bincount(yb)}")

        if self.regen_data or not os.path.exists(self.scaler_path):
            self.scaler = Pipeline([
                # ("std", StandardScaler()),
                ("mm",  MinMaxScaler(feature_range=(0, 1)))
            ])
            Xs = self.scaler.fit_transform(Xb)
            with open(self.scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
        else:
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            Xs = self.scaler.transform(Xb)

        # 3) cast to float32 / int32
        return Xs.astype(np.float32), yb.astype(np.int32)

    def _make_sequences(self, X: np.ndarray, y: np.ndarray) -> tf.data.Dataset:
        """
        Build a tf.data pipeline that:
          - creates a sliding window of length seq_len
          - batches those windows into (batch, seq_len, features)
          - one-hot encodes the labels
        without ever stacking all windows in RAM.
        """
        seq_len = self.seq_len
        num_classes = self.num_classes

        # 1) base dataset of single timesteps
        ds = tf.data.Dataset.from_tensor_slices((X, y))

        # 2) sliding windows of length seq_len
        ds = ds.window(size=seq_len, shift=1, drop_remainder=True)

        # 3) turn each “window” dataset into a (seq_len, features), (seq_len,) tuple
        ds = ds.flat_map(
            lambda x, y: tf.data.Dataset.zip((
                x.batch(seq_len, drop_remainder=True),
                y.batch(seq_len, drop_remainder=True),
            ))
        )

        # 4) one‐hot encode the labels and batch
        ds = ds.map(lambda x, y: (x, tf.one_hot(y, depth=num_classes)),
                    num_parallel_calls=tf.data.AUTOTUNE)

        # 5) shuffle & batch for training
        return ds.shuffle(10_000).batch(32).prefetch(tf.data.AUTOTUNE)

    def prepare_datasets(self):
        # Train
        X_train, y_train = self._get_flat_xy(self.train_content)
        Xt, yt = self._balance_and_scale(X_train, y_train, "Train")
        self.train_ds = self._make_sequences(Xt, yt)

        # Validation
        X_val, y_val = self._get_flat_xy(self.val_content)
        Xv, yv = self._balance_and_scale(X_val, y_val, "Val")
        self.val_ds = self._make_sequences(Xv, yv)

        # Test  (no SMOTE on test—just scale & sequence)
        X_test, y_test = self._get_flat_xy(self.test_content)
        Xtst = self.scaler.transform(X_test)
        self.test_ds = self._make_sequences(Xtst, y_test)

    def tune(self, max_trials: int = 20):
        self.prepare_datasets()

        # 4) Create a distribution strategy
        strategy = tf.distribute.MirroredStrategy()
        print("Number of replicas:", strategy.num_replicas_in_sync)

        with strategy.scope():
            hypermodel = SequenceClassifierHyperModel(
                seq_len=self.seq_len,
                num_features=next(iter(self.train_ds))[0].shape[-1],
                num_classes=self.num_classes
            )
            tuner = kt.RandomSearch(
                hypermodel,
                objective="val_accuracy",
                max_trials=max_trials,
                directory=os.path.join(self.cache_dir, "tuner"),
                project_name="tf_seq",

            )
        # 5) Run search as usual
        tuner.search(
            self.train_ds,
            epochs=10,
            validation_data=self.val_ds,
            steps_per_epoch=200,
        )

        # 6) After tuning, rebuild the best model under the same strategy
        with strategy.scope():
            self.best_hp = tuner.get_best_hyperparameters(1)[0]
            self.model = tuner.hypermodel.build(self.best_hp)

    def train(self, epochs: int = 30):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            if not hasattr(self, "model"):
                hp = kt.HyperParameters()
                self.model = SequenceClassifierHyperModel(
                    seq_len=self.seq_len,
                    num_features=next(iter(self.train_ds))[0].shape[-1],
                    num_classes=self.num_classes
                ).build(hp)

            self.model.compile(  # re-compile inside the scope if needed
                optimizer=self.model.optimizer,
                loss=self.model.loss,
                metrics=self.model.metrics,
            )

        self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            steps_per_epoch=200,
        )

    def evaluate(self):
        results = self.model.evaluate(self.test_ds)
        logging.info(f"Test results: {results}")
        return results


if __name__ == "__main__":
    trainer = TFSequenceTrainer(
        classes=["NO_POI", "POI1", "POI2", "POI3", "POI4", "POI5", "POI6"],
        seq_len=1000,
        regen_data=True
    )
    trainer.tune(max_trials=10)
    trainer.train(epochs=20)
    trainer.evaluate()
