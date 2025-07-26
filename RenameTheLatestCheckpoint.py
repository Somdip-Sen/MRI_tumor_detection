import os
import glob
import re


def archive_latest_run(checkpoint_dir='./Checkpoints'):
    """
    Finds the latest version number from saved models in the checkpoint directory
    and renames the current un-versioned files to the next version number.
    """
    print(f"--- Archiving latest run in '{checkpoint_dir}' ---")

    if not os.path.isdir(checkpoint_dir):
        print(f"Error: Checkpoint directory '{checkpoint_dir}' not found.")
        return

    # 1. Define the base filenames we want to archive
    base_model_name = 'brain_tumor_classifier_best.pth'
    base_checkpoint_name = 'training_checkpoint.pth'

    # 2. Find the highest existing version number (X)
    # We'll search for the pattern '...bestX.pth' to determine the last run number
    search_pattern = os.path.join(checkpoint_dir, 'brain_tumor_classifier_best*.pth')
    all_model_files = glob.glob(search_pattern)

    max_version = 0
    for file_path in all_model_files:
        # A regular expression to find numbers in a filename like '...best1.pth'
        match = re.search(r'best(\d+)\.pth', file_path)
        if match:
            version = int(match.group(1))
            if version > max_version:
                max_version = version

    new_version = max_version + 1
    print(f"Found max existing version: {max_version}. New version will be: {new_version}")

    # 3. Define the new filenames for the archive
    new_model_name = f'brain_tumor_classifier_best{new_version}.pth'
    new_checkpoint_name = f'training_checkpoint{new_version}.pth'

    # 4. Construct the full paths for the source and destination files
    source_model_path = os.path.join(checkpoint_dir, base_model_name)
    dest_model_path = os.path.join(checkpoint_dir, new_model_name)

    source_checkpoint_path = os.path.join(checkpoint_dir, base_checkpoint_name)
    dest_checkpoint_path = os.path.join(checkpoint_dir, new_checkpoint_name)

    # 5. Rename the best model file, if it exists
    if os.path.exists(source_model_path):
        os.rename(source_model_path, dest_model_path)
        print(f"Renamed '{base_model_name}' to '{new_model_name}'")
    else:
        print(f"No base model file ('{base_model_name}') found to archive.")

    # 6. Rename the main checkpoint file, if it exists
    if os.path.exists(source_checkpoint_path):
        os.rename(source_checkpoint_path, dest_checkpoint_path)
        print(f"Renamed '{base_checkpoint_name}' to '{new_checkpoint_name}'")
    else:
        print(f"No base checkpoint file ('{base_checkpoint_name}') found to archive.")

    print("--- Archiving complete --- \n")


# To run the script from your terminal or another file
if __name__ == '__main__':
    archive_latest_run()