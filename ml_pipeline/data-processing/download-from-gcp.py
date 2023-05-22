from google.cloud import storage

storage_client = storage.Client(project="home-assistant-pb96")

bucket_name = 'home-assistant-pb96-tapo'

destination_folder = "data/all-tapo-20230312"

# download all files in bucket
blobs = storage_client.list_blobs(bucket_name)
total_size = 0
for blob in blobs:
    # print(blob.name)
    # print(blob.size)
    total_size += blob.size
    # blob.download_to_filename(f"{destination_folder}/{blob.name}")

print(total_size / 1000000)
