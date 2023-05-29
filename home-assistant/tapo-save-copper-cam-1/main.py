# from pytapo import Tapo
from google.cloud import storage as gcs
from google.oauth2 import service_account, id_token
import requests
import google.auth.transport.requests
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
import shutil
from datetime import datetime
import tempfile
from configparser import ConfigParser

config = ConfigParser()
config.read("../../.config/config.ini")


def main(event, context):
    # user = "copper_cam_pb" # user you set in Advanced Settings ->
    # Camera Account
    # password = "petekaren" # password you set in Advanced Settings ->
    # Camera Account
    # host = "10.0.0.10" # ip of the camera, example: 192.168.1.52

    disable_warnings(InsecureRequestWarning)

    tmpdir = tempfile.gettempdir()

    pat = config.get('home-assistant', 'pat')

    url = config.get('home-assistant', 'url')

    headers = {
        "Authorization": f"Bearer {pat}",
        "content-type": "application/json",
    }

    gcs_cred_file = "home-assistant-pb96-5987a3dcb0dd.json"
    gcs_creds = service_account.Credentials.from_service_account_file(
        gcs_cred_file
    )
    gcs_client = gcs.Client(credentials=gcs_creds)

    timestamp = str(int(round(datetime.utcnow().timestamp(), 0)))
    response = requests.get(url, headers=headers, stream=True, verify=False)
    with open(tmpdir + "/img.jpg", "wb") as out_file:
        shutil.copyfileobj(response.raw, out_file)
    print("Home Assistant fetch image: " + str(response.status_code))

    bucket = gcs_client.get_bucket("home-assistant-pb96-tapo")
    new_blob = bucket.blob("copper_cam_sd_stream" + "_" + timestamp)
    new_blob.upload_from_filename(tmpdir + "/img.jpg")
    print("GCS upload complete")

    model_url = config.get('home-assistant', 'model_url')

    # cloud_creds = Credentials.from_authorized_user_info(info=None)
    request = google.auth.transport.requests.Request()
    cloud_cred_token = id_token.fetch_id_token(request, model_url)

    my_img = {'image': open(tmpdir + "/img.jpg", 'rb')}
    headers = {'Authorization': f'Bearer {cloud_cred_token}'}
    r = requests.post(model_url, files=my_img, headers=headers, verify=False)
    print("Model response: ", r.text)

    payload = {
        "message": str(r.text)
    }
    print(payload)

    ha_webhook_url = config.get('home-assistant', 'ha_webhook_url') + "="
    ha_webhook_response = requests.post(
        ha_webhook_url,
        headers={
            "content-type": "application/json",
            "Accept": "*/*",
            "Connection": "keep-alive"
        },
        json=payload
    )
    print("HA webhook: " +
          str(ha_webhook_response.status_code))


try:
    main(None, None)
except Exception as e:
    print(e)
