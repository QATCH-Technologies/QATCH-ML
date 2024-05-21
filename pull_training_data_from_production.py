import os
import shutil

roots = []
paths = []
files = []

def iterate_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            # print(os.path.join(root, file))
            if file.endswith("_poi.csv") or file == "capture.zip":
                store_data_file(root, file)

def store_data_file(root, file):
    roots.append(root)
    paths.append(os.path.join(root, file))

def process_stored_files():
    for x,y in enumerate(roots):
        if roots.count(y) == 2:
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
            unzip_loc = os.path.join(os.path.dirname(f), "capture")
            if os.path.exists(unzip_loc):
                zf = os.listdir(unzip_loc)
                for _f in zf:
                    if _f.endswith(".csv") and _f.find("_lower.csv") == -1 and _f.find("_tec.csv") == -1:
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
                        if _f.endswith(".csv") and _f.find("_lower.csv") == -1 and _f.find("_tec.csv") == -1:
                            # unzip_required = False
                            unzipped_data_path = os.path.join(unzip_loc, _f)
                            break
            f = unzipped_data_path
        src = f
        dst = os.path.join(destination_path, os.path.basename(f))
        if os.path.exists(src) and not os.path.exists(dst):
            print("Copying:", os.path.basename(f))
            shutil.copyfile(src, dst)

destination_path = r"C:\Users\Alexander J. Ross\Documents\QATCH Work\ML\content\testing-2024-02-27"
iterate_files(r"C:\Users\Alexander J. Ross\Documents\QATCH Work\ML\content\testing-2024-02-27")
process_stored_files()


