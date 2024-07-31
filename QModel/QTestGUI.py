import tkinter as tk
from tkinter import filedialog, messagebox
from QTest import *


# Placeholder functions for running tests and generating reports
def run_tests(directory):
    # Replace with actual test-running logic
    qmp_list, md_list, qmp_deltas, md_deltas = run()
    return qmp_list, md_list


def generate_report(report_type, qmp_list, md_list):
    # Replace with actual report-generating logic
    if report_type == "QMP Report":
        return f"Generated QMP Report: {qmp_list}"
    elif report_type == "MD Report":
        return f"Generated MD Report: {md_list}"
    else:
        return "Invalid report type"


class TestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Test GUI")

        self.directory = ""
        self.qmp_list = []
        self.md_list = []

        self.create_widgets()

    def create_widgets(self):
        self.select_dir_button = tk.Button(
            self.root, text="Select Test Directory", command=self.select_directory
        )
        self.select_dir_button.pack(pady=10)

        self.run_tests_button = tk.Button(
            self.root, text="Run Tests", command=self.run_tests
        )
        self.run_tests_button.pack(pady=10)

        self.report_type_label = tk.Label(self.root, text="Select Report Type:")
        self.report_type_label.pack(pady=10)

        self.report_type_var = tk.StringVar(value="QMP Report")
        self.qmp_report_radio = tk.Radiobutton(
            self.root,
            text="QMP Report",
            variable=self.report_type_var,
            value="QMP Report",
        )
        self.qmp_report_radio.pack(pady=5)
        self.md_report_radio = tk.Radiobutton(
            self.root,
            text="MD Report",
            variable=self.report_type_var,
            value="MD Report",
        )
        self.md_report_radio.pack(pady=5)

        self.generate_report_button = tk.Button(
            self.root, text="Generate Report", command=self.generate_report
        )
        self.generate_report_button.pack(pady=10)

    def select_directory(self):
        self.directory = filedialog.askdirectory()
        if self.directory:
            messagebox.showinfo(
                "Directory Selected", f"Selected directory: {self.directory}"
            )

    def run_tests(self):
        if not self.directory:
            messagebox.showwarning(
                "No Directory Selected", "Please select a directory first."
            )
            return

        self.qmp_list, self.md_list = run_tests(self.directory)
        messagebox.showinfo("Tests Completed", "Tests have been run successfully.")

    def generate_report(self):
        if not self.qmp_list or not self.md_list:
            messagebox.showwarning("No Test Data", "Please run the tests first.")
            return

        report_type = self.report_type_var.get()
        report = generate_report(report_type, self.qmp_list, self.md_list)
        messagebox.showinfo("Report Generated", report)


if __name__ == "__main__":
    root = tk.Tk()
    app = TestGUI(root)
    root.mainloop()
