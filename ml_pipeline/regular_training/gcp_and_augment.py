from google.cloud import storage
from data_processing import x2_augment as aug
import datetime
import os


# download all the files in the bucket with the prefix main-dataset
def download_from_gcp_all():
    storage_client = storage.Client(project="home-assistant-pb96")

    bucket_name = 'home-assistant-pb96-tapo'

    destination_folder = f"data/all-main-gcp-{datetime.date.today().strftime('%Y%m%d')}"
    os.makedirs(destination_folder, exist_ok=True)

    for name in ["bed", "missing", "rug", "somewhere"]:
        os.makedirs(f"{destination_folder}/class_{name}", exist_ok=True)


    # download all files in bucket
    blobs = storage_client.list_blobs(bucket_name)
    total_size = 0
    for blob in blobs:
        # print(blob.name)
        # print(blob.size)
        total_size += blob.size
        # blob.download_to_filename(f"{destination_folder}/{blob.name}")

    print(total_size / 1000000)

    # download all the files in the bucket with the prefix main-dataset
    blobs = storage_client.list_blobs(bucket_name, prefix="main-dataset/")
    download_progress = 0
    previous_download_progress = 0
    for blob in blobs:
        name = "/".join(blob.name.split("/")[1:])
        download_progress += blob.size
        if int(download_progress) != previous_download_progress:
            print(download_progress)
            previous_download_progress = int(download_progress)
        blob.download_to_filename(f"{destination_folder}/{name}")
    
    return destination_folder


# balance and augment x2 the downloaded files
def augment_download_2x(source_folder: str):
    destination_folder = f"data/all-main-gcp-augmented2x-{datetime.date.today().strftime('%Y%m%d')}"
    aug.augment_2x(source_folder, destination_folder)
    return destination_folder
