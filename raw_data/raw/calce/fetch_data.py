## fecth data from calcle: https://calce.umd.edu/battery-data

import os
import requests
import zipfile
from io import BytesIO


def fetch_calce_data(save_dir: str = "data/raw/calce") -> None:
    """
    Fetches the CALCE battery dataset and saves it to the specified directory.

    Args:
        save_dir: Directory where the dataset will be saved. Default is "data/raw/calce".
    """
    os.makedirs(save_dir, exist_ok=True)

    # URL of the CALCE battery dataset (replace with actual URL if available)
    url = "https://calce.umd.edu/sites/calce.umd.edu/files/battery_data.zip"

    print(f"Downloading CALCE battery dataset from {url}...")
    response = requests.get(url)

    if response.status_code == 200:
        print("Download successful. Extracting files...")
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(save_dir)
        print(f"Dataset extracted to {save_dir}.")
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")
