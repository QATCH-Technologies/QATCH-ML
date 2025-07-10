import tensorflow as tf
import keras_tuner as kt
from keras import layers, Model


class SequenceClassifierHyperModel(kt.HyperModel):
    def __init__(self, seq_len, num_features, num_classes=7):
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_classes = num_classes

    def build(self, hp):
        inp = layers.Input(
            shape=(self.seq_len, self.num_features), name="signal_in")
        x = inp

        # ---- Tunable stack of RNN layers ----
        for i in range(hp.Int("rnn_layers", 1, 3, default=2)):
            units = hp.Int(f"units_{i}", 32, 256, step=32, default=64)
            rnn_type = hp.Choice(
                f"rnn_type_{i}", ["LSTM", "GRU"], default="GRU")
            return_seq = True  # we need timestep-wise outputs
            if rnn_type == "LSTM":
                layer = layers.LSTM(units, return_sequences=return_seq)
            else:
                layer = layers.GRU(units, return_sequences=return_seq)
            if hp.Boolean(f"bidirectional_{i}", default=True):
                layer = layers.Bidirectional(layer)
            x = layer(x)
            x = layers.Dropout(
                hp.Float(f"dropout_{i}", 0.0, 0.5, step=0.1, default=0.2)
            )(x)

        # ---- Tunable time-distributed bottleneck ----
        dense_units = hp.Int("dense_units", 16, 128, step=16, default=32)
        x = layers.TimeDistributed(layers.Dense(
            dense_units, activation="relu"))(x)

        # ---- Final softmax over 7 classes (0–6) per timestep ----
        out = layers.TimeDistributed(
            layers.Dense(self.num_classes, activation="softmax"), name="output"
        )(x)

        model = Model(inp, out)

        # ---- Tunable optimizer LR ----
        lr = hp.Float("learning_rate", 1e-4, 1e-2,
                      sampling="log", default=1e-3)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


# --- Example of wiring up the tuner ---
# seq_len, num_features need to be set to your data dims:
hypermodel = SequenceClassifierHyperModel(seq_len=1000, num_features=10)

tuner = kt.RandomSearch(
    hypermodel,
    objective="val_accuracy",
    max_trials=20,
    executions_per_trial=1,
    directory="tuner_logs",
    project_name="seq_class",
)

# Then run:
# tuner.search(x_train, y_train,
#              epochs=20,
#              validation_data=(x_val, y_val),
#              class_weight=your_class_weights)
