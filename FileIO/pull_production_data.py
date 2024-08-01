import os
import shutil
import numpy as np

roots = []
paths = []
files = []

num_files_per_run = 3


def iterate_files(directory):
    for root, dirs, files in os.walk(directory):
        file_poi = None
        file_xml = None
        file_zip = None
        for file in files:
            # print(os.path.join(root, file))
            if file.endswith("_poi.csv"):
                file_poi = file
            if file.endswith(".xml"):
                with open(os.path.join(root, file), "r") as f:
                    xml_text = "\n".join(f.readlines()).upper()
                    if "MM240506" in xml_text:
                        print(f"Skipping {file}... bad batch 1")
                        continue
                    if "DD240501" in xml_text:
                        print(f"Skipping {file}... bad batch 2")
                        continue
                file_xml = file
            if file == "capture.zip":
                file_zip = file
        if all([file_poi, file_xml, file_zip]):
            store_data_file(root, file_poi)
            store_data_file(root, file_xml)
            store_data_file(root, file_zip)


def store_data_file(root, file):
    i = int(len(roots) / num_files_per_run)
    copy_from = os.path.join(root, file)
    copy_to = os.path.join(training_data_path, f"{i:05.0f}")

    try:
        os.makedirs(copy_to, exist_ok=True)

        root = copy_to
        roots.append(root)

        if os.path.isfile(os.path.join(copy_to, file)):
            paths.append(os.path.join(copy_to, file))
            raise FileExistsError()

        print(f"Copying {i:05.0f}:", os.path.basename(copy_from))
        file = shutil.copy(copy_from, copy_to)
        paths.append(file)

    except:
        print(f"Skipping {i:05.0f}/{file}... file already exists!")


def process_stored_files():
    for x, y in enumerate(roots):
        if roots.count(y) == num_files_per_run:
            # print(paths[x])
            files.append(paths[x])
    skip_next = False
    for f in files:
        if skip_next:
            skip_next = False
            continue
        if f.endswith("capture.zip"):
            # check if already unzipped
            unzip_required = True
            unzipped_data_path = f
            unzip_loc = os.path.dirname(
                f
            )  # os.path.join(os.path.dirname(f), "capture")
            if os.path.exists(unzip_loc) and False:  # disable for new code
                zf = os.listdir(unzip_loc)
                for _f in zf:
                    if (
                        _f.endswith(".csv")
                        and _f.find("_lower.csv") == -1
                        and _f.find("_tec.csv") == -1
                    ):
                        unzip_required = False
                        unzipped_data_path = os.path.join(unzip_loc, _f)
                        break
            if unzip_required:
                print("Extracting:", os.path.basename(os.path.dirname(f)))
                try:
                    shutil.unpack_archive(f, unzip_loc)
                except RuntimeError as re:
                    print(re)
                    skip_next = True
                    continue
                if os.path.exists(unzip_loc):
                    zf = os.listdir(unzip_loc)
                    for _f in zf:
                        if (
                            _f.endswith(".csv")
                            and _f.find("_lower.csv") == -1
                            and _f.find("_tec.csv") == -1
                        ):
                            # unzip_required = False
                            unzipped_data_path = os.path.join(unzip_loc, _f)
                            os.remove(
                                f
                            )  # delete ZIP if data file found; otherwise leave capture.zip for review
                            break
            f = unzipped_data_path
        if False:  # do not copy again, old code
            src = f
            dst = os.path.join(training_data_path, os.path.basename(f))
            if os.path.exists(src) and not os.path.exists(dst):
                print("Copying:", os.path.basename(f))
                shutil.copyfile(src, dst)
    for root in np.unique(roots):
        for file in os.listdir(root):
            full_path = os.path.join(root, file)
            if full_path.endswith(".crc") or full_path.endswith("_tec.csv"):
                print("Purging extra files:", file)
                os.remove(full_path)  # delete extra files


production_data_path = (
    r"C:\Users\QATCH\QATCH Dropbox\QATCH Team Folder\Production Notes"
)
training_data_path = r"C:\Users\QATCH\dev\clean\QATCH-ML\content\dropbox_dump"

iterate_files(production_data_path)
process_stored_files()


def copy_files(src_dir, dest_dir):
    # Ensure destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # List all files in the source directory
    files = os.listdir(src_dir)

    # Copy each file to the destination directory
    for file in files:
        src_file = os.path.join(src_dir, file)
        dest_file = os.path.join(dest_dir, file)
        shutil.copy2(src_file, dest_file)
        print(f"Copied {src_file} to {dest_file}")


# Define source and destination directories
src_directory = "content/dropbox_dump"
dest_directory = "content/all_data"

# Call the function to copy files
copy_files(src_directory, dest_directory)
