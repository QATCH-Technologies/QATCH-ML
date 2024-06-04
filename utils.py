import time
import matplotlib.pyplot as plt


def linebreak():
    print(
        "\n---------------------------------------------------------------------------------------------------------------------\n"
    )


def loading(info):
    animation_sequence = "|/-\\"
    idx = 0
    while True:
        print(animation_sequence[idx % len(animation_sequence)], end="\r")
        idx += 1
        time.sleep(0.1)

        if idx == len(animation_sequence):
            idx = 0

        # Verify the change in idx variable
        print(f" {info}", end="\r")


def status(message):
    print("(status)", message)


def error(message):
    print("(err)", message)


def echo(message):
    print("(echo)", message)


def info(message):
    print("(info)", message)


def plot_loss(history):
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, "r", label="Training Loss")
    plt.plot(epochs, val_loss, "b", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
