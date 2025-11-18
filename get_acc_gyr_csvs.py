import os
import shutil
import zipfile


# Input and output dirs
DATASET_DIR = "dataset/raw_all"
OUTPUT_DIR = "dataset/raw_acc_gyr"

# Sensor prefixes we care about
ACC_PREFIXES = ["acc_"]  # lowercase match
GYR_PREFIXES = ["gyr_", "gyroscope_"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def is_zip(filename):
    return filename.lower().endswith("csv.zip")


def extract_from_zip_object(zip_obj, zip_name, out_dir, level=0):
    """
    Extracts forearm accelerometer and gyroscope CSV files from a zip object.
    If nested ZIPs are found, this function processes them recursively.
    """

    indent = "  " * level
    print(f"{indent}Scanning ZIP: {zip_name}")

    extracted = []

    for member in zip_obj.namelist():
        member_lower = member.lower()

        # Case 1: Normal CSV files in the ZIP
        if member_lower.endswith("_forearm.csv"):

            if any(member_lower.startswith(p) for p in ACC_PREFIXES + GYR_PREFIXES):
                print(f"{indent}  -> Found target CSV: {member}")
                target_path = os.path.join(out_dir, os.path.basename(member))

                # Extract temporarily
                zip_obj.extract(member, out_dir)

                # Flatten if extracted into a subfolder
                extracted_path = os.path.join(out_dir, member)
                if os.path.exists(extracted_path):
                    shutil.move(extracted_path, target_path)
                extracted.append(target_path)

        # Case 2: Nested ZIP files inside ZIP → process recursively
        elif is_zip(member):
            print(f"{indent}  -> Found nested ZIP: {member}")

            # Extract nested ZIP to memory
            nested_bytes = zip_obj.read(member)
            nested_zip_path = os.path.join(out_dir, "tmp_nested.zip")

            # Write to temporary file
            with open(nested_zip_path, "wb") as f:
                f.write(nested_bytes)

            # Process nested ZIP
            with zipfile.ZipFile(nested_zip_path, "r") as nested_zip:
                extracted_nested = extract_from_zip_object(
                    nested_zip, zip_name + " -> " + member, out_dir, level=level + 1
                )
                extracted.extend(extracted_nested)

            # Remove temporary nested ZIP
            os.remove(nested_zip_path)

    return extracted


def process_participant(proband_path, proband_name):
    print()
    print("==============================")
    print(f"Processing {proband_name}")
    print("==============================")

    data_dir = os.path.join(proband_path, "data")
    if not os.path.isdir(data_dir):
        print(f"Skipping {proband_name}: no data/ folder.")
        return
    
    subject_id = proband_name.replace("proband","")
    proband_name = subject_id + "proband"

    # Output dirs
    acc_out = os.path.join(OUTPUT_DIR, proband_name, "acc")
    gyr_out = os.path.join(OUTPUT_DIR, proband_name, "gyr")
    ensure_dir(acc_out)
    ensure_dir(gyr_out)

    for zip_file in sorted(os.listdir(data_dir)):
        if not is_zip(zip_file) or not zip_file.startswith(("acc_", "gyr_")):
            continue

        zip_path = os.path.join(data_dir, zip_file)
        print(f"\nOpening ZIP: {zip_path}")

        # Extraction directory for this zip
        tmp_extract_dir = os.path.join(OUTPUT_DIR, proband_name, "tmp_extracted")
        ensure_dir(tmp_extract_dir)

        with zipfile.ZipFile(zip_path, "r") as z:
            extracted_files = extract_from_zip_object(z, zip_file, tmp_extract_dir)

        # Move categorized files to acc/ or gyr/
        for f in extracted_files:
            fname = os.path.basename(f).lower()
            if any(fname.startswith(p) for p in ACC_PREFIXES):
                print(f"  Moving {fname} -> acc/")
                shutil.move(f, os.path.join(acc_out, fname))
            elif any(fname.startswith(p) for p in GYR_PREFIXES):
                print(f"  Moving {fname} -> gyr/")
                shutil.move(f, os.path.join(gyr_out, fname))

        # Clean up temporary extraction dir
        shutil.rmtree(tmp_extract_dir, ignore_errors=True)


def process_all():
    ensure_dir(OUTPUT_DIR)
    participants = sorted(
        [d for d in os.listdir(DATASET_DIR) if d.startswith("proband")]
    )

    print(f"Found participants: {participants}")
    for p in participants:
        process_participant(os.path.join(DATASET_DIR, p), p)

    print("\nAll participants processed!")
    print(f"Check the '{OUTPUT_DIR}/' folder.")


if __name__ == "__main__":
    process_all()
