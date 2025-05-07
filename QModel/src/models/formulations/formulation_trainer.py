import sys
import pickle
import pandas as pd
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

# Feature definitions (must match training)
CATEGORICAL_FEATURES = ['Protein type', 'Buffer', 'Sugar', 'Surfactant']
NUMERIC_FEATURES = ['Protein', 'Temperature', 'Sugar (M)', 'TWEEN']
VISCOUS_COLUMNS = [
    'Viscosity @ 100',
    'Viscosity  @ 1000',
    'Viscosity  @ 10000',
    'Viscosity  @ 100000',
    'Viscosity @ 15000000',
]
DEFAULT_MODEL_PATH = 'viscosity_model.pkl'


class ViscosityPredictorGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Viscosity Profile Predictor')
        self.df = None
        self.X = None
        self.y = None
        self._load_model()
        self._init_ui()

    def _build_pipeline(self):
        # Create preprocessing + model pipeline
        preprocessor = ColumnTransformer([
            ('onehot', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
            ('scaler', StandardScaler(),                   NUMERIC_FEATURES),
        ])
        return Pipeline([
            ('preproc', preprocessor),
            ('regressor', MultiOutputRegressor(
                XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    objective='reg:squarederror',
                    random_state=42
                )
            )),
        ])

    def _load_model(self):
        # Try loading an existing model, or build a fresh pipeline
        try:
            with open(DEFAULT_MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
        except Exception:
            self.model = self._build_pipeline()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout()

        # Dataset & training controls
        btn_layout = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton('Load Dataset')
        self.load_btn.clicked.connect(self._on_load_dataset)
        btn_layout.addWidget(self.load_btn)

        self.train_btn = QtWidgets.QPushButton('Train Model')
        self.train_btn.clicked.connect(self._on_train_model)
        self.train_btn.setEnabled(False)
        btn_layout.addWidget(self.train_btn)

        self.pred_dataset_btn = QtWidgets.QPushButton('Predict on Dataset')
        self.pred_dataset_btn.clicked.connect(self._on_predict_dataset)
        self.pred_dataset_btn.setEnabled(False)
        btn_layout.addWidget(self.pred_dataset_btn)

        layout.addLayout(btn_layout)

        # Single‐sample prediction form
        form = QtWidgets.QFormLayout()
        self.inputs = {}
        # Populate categorical combos from pipeline definition (if available)
        try:
            preproc = self.model.named_steps['preproc']
            onehot = preproc.named_transformers_['onehot']
            categories = onehot.categories_
        except Exception:
            categories = [[] for _ in CATEGORICAL_FEATURES]

        for feat, cats in zip(CATEGORICAL_FEATURES, categories):
            combo = QtWidgets.QComboBox()
            combo.addItems([str(c) for c in cats])
            form.addRow(feat + ':', combo)
            self.inputs[feat] = combo

        # Numeric inputs
        for feat in NUMERIC_FEATURES:
            line = QtWidgets.QLineEdit()
            line.setValidator(QDoubleValidator(0.0, 1e12, 6))
            form.addRow(feat + ':', line)
            self.inputs[feat] = line

        layout.addLayout(form)

        # Predict button for single inputs
        self.predict_btn = QtWidgets.QPushButton('Predict Single')
        self.predict_btn.clicked.connect(self._on_predict)
        layout.addWidget(self.predict_btn)

        # Results display
        result_group = QtWidgets.QGroupBox('Predicted Viscosity Profile')
        result_layout = QtWidgets.QFormLayout()
        self.result_labels = {}
        for col in VISCOUS_COLUMNS:
            lbl = QtWidgets.QLabel('-')
            result_layout.addRow(col + ':', lbl)
            self.result_labels[col] = lbl
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        self.setLayout(layout)

    def _on_load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open CSV', '', 'CSV Files (*.csv)')
        if not path:
            return
        try:
            df = pd.read_csv(path)
            required = set(CATEGORICAL_FEATURES +
                           NUMERIC_FEATURES + VISCOUS_COLUMNS)
            if not required.issubset(df.columns):
                missing = required - set(df.columns)
                QMessageBox.warning(self, 'Invalid Data',
                                    f'Missing columns: {missing}')
                return
            self.df = df
            self.X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
            self.y = df[VISCOUS_COLUMNS]
            self.train_btn.setEnabled(True)
            QMessageBox.information(
                self, 'Dataset Loaded', f'Loaded {len(df)} rows.')
        except Exception as e:
            QMessageBox.warning(self, 'Load Error', str(e))

    def _on_train_model(self):
        if self.df is None:
            QMessageBox.warning(self, 'No Data', 'Load a dataset first.')
            return
        try:
            self.model = self._build_pipeline()
            self.model.fit(self.X, self.y)
            save_path, _ = QFileDialog.getSaveFileName(
                self, 'Save Model', DEFAULT_MODEL_PATH, 'Pickle (*.pkl)')
            if save_path:
                with open(save_path, 'wb') as f:
                    pickle.dump(self.model, f)
            self.pred_dataset_btn.setEnabled(True)
            QMessageBox.information(
                self, 'Training Complete', 'Model trained successfully.')
        except Exception as e:
            QMessageBox.warning(self, 'Training Error', str(e))

    def _on_predict_dataset(self):
        if self.df is None:
            QMessageBox.warning(self, 'No Data', 'Load and train model first.')
            return
        try:
            preds = self.model.predict(self.X)
            df_out = self.df.copy()
            for i, col in enumerate(VISCOUS_COLUMNS):
                df_out[col] = preds[:, i]
            save_path, _ = QFileDialog.getSaveFileName(
                self, 'Save Predictions', 'predictions.csv', 'CSV Files (*.csv)')
            if save_path:
                df_out.to_csv(save_path, index=False)
                QMessageBox.information(
                    self, 'Saved', f'Predictions saved to {save_path}')
        except Exception as e:
            QMessageBox.warning(self, 'Prediction Error', str(e))

    def _on_predict(self):
        try:
            data = {f: [self.inputs[f].currentText()]
                    for f in CATEGORICAL_FEATURES}
            for f in NUMERIC_FEATURES:
                txt = self.inputs[f].text()
                if txt == '':
                    raise ValueError(f'Missing value for {f}')
                data[f] = [float(txt)]
            df = pd.DataFrame(data)
            preds = self.model.predict(df)[0]
            for col, val in zip(VISCOUS_COLUMNS, preds):
                self.result_labels[col].setText(f'{val:.3f}')
        except Exception as e:
            QMessageBox.warning(self, 'Prediction Error', str(e))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ViscosityPredictorGUI()
    window.show()
    sys.exit(app.exec_())
