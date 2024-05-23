import time


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
