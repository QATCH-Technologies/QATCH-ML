import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import random


class CSVSorterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Sorter")

        self.csv_files = []
        self.current_file_index = -1

        self.good_dir = filedialog.askdirectory(title="Select Good Directory")
        self.bad_dir = filedialog.askdirectory(title="Select Bad Directory")
        self.source_dir = filedialog.askdirectory(title="Select Source Directory")

        self.load_csv_files()

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.good_button = tk.Button(
            self.button_frame, text="Good", command=self.mark_good
        )
        self.good_button.pack(side=tk.LEFT)

        self.bad_button = tk.Button(
            self.button_frame, text="Bad", command=self.mark_bad
        )
        self.bad_button.pack(side=tk.LEFT)

        self.next_file()

    def load_csv_files(self):
        content = []
        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                content.append(os.path.join(root, file))

        for filename in content:
            if (
                filename.endswith(".csv")
                and not filename.endswith("_poi.csv")
                and not file.endswith("_lower.csv")
            ):
                self.csv_files.append(filename)
        random.shuffle(self.csv_files)
        self.csv_files.sort()

    def plot_csv(self, filepath):
        df = pd.read_csv(filepath)
        poi = pd.read_csv(filepath.replace(".csv", "_poi.csv"), header=None).values
        self.ax.clear()
        self.ax.plot(df["Dissipation"])
        self.ax.axvline(
            x=poi[0],
            color="grey",
            linestyle="dashed",
            label=f"Actual POI {1}",
        )
        for i, index in enumerate(poi):
            self.ax.axvline(
                x=index,
                color="grey",
                linestyle="dashed",
                label=f"Actual POI {i + 1}",
            )
        self.ax.annotate(
            poi,
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"),
        )
        self.ax.set_title(os.path.basename(filepath))
        self.canvas.draw()

    def next_file(self):
        self.current_file_index += 1
        if self.current_file_index < len(self.csv_files):
            current_file_path = os.path.join(
                self.source_dir, self.csv_files[self.current_file_index]
            )
            self.plot_csv(current_file_path)
        else:
            tk.messagebox.showinfo("Done", "No more files to review")
            self.root.quit()

    def mark_good(self):
        current_file_path = os.path.join(
            self.source_dir, self.csv_files[self.current_file_index]
        )
        shutil.copy(current_file_path, self.good_dir)
        shutil.copy(current_file_path.replace(".csv", "_poi.csv"), self.good_dir)
        self.next_file()

    def mark_bad(self):
        current_file_path = os.path.join(
            self.source_dir, self.csv_files[self.current_file_index]
        )
        shutil.copy(current_file_path, self.bad_dir)
        shutil.copy(current_file_path.replace(".csv", "_poi.csv"), self.bad_dir)
        self.next_file()


if __name__ == "__main__":
    root = tk.Tk()
    app = CSVSorterApp(root)
    root.mainloop()
