from google.cloud import storage
import time, datetime, os

storage_client = storage.Client(project="home-assistant-pb96")

bucket_name = 'home-assistant-pb96-tapo'

destination_folder = f"data/last-day-tapo-{datetime.date.today().strftime('%Y%m%d')}"
os.makedirs(destination_folder, exist_ok=True)

# find size of the last day's files in bucket
blobs = storage_client.list_blobs(bucket_name, start_offset=f"copper_cam_sd_stream_{time.time() - 1 * 24 * 60 * 60}")
total_size = 0
for blob in blobs:
    # print(blob.name)
    # print(blob.size)
    total_size += blob.size

print(total_size / 1000000)

# download the last day's files in bucket
blobs = storage_client.list_blobs(bucket_name, start_offset=f"copper_cam_sd_stream_{time.time() - 1 * 24 * 60 * 60}")
download_progress = 0
previous_download_progress = 0
for blob in blobs:
    download_progress += blob.size
    if int(download_progress / 1000000) != previous_download_progress:
        print(download_progress / 1000000)
        previous_download_progress = int(download_progress / 1000000)
    blob.download_to_filename(f"{destination_folder}/{blob.name}")
