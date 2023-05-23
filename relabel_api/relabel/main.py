import functions_framework
# import tempfile
import flask
import json
from datetime import datetime
from google.cloud import storage


@functions_framework.http
def main(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    Note:
        For more information on how Flask integrates with Cloud
        Functions, see the `Writing HTTP functions` page.
        <https://cloud.google.com/functions/docs/writing/http#http_frameworks>
    """

    # get the os tempdir
    # tmpdir = tempfile.gettempdir()

    # create google cloud storage client
    storage_client = storage.Client(project="home-assistant-pb96")
    bucket = storage_client.get_bucket("home-assistant-pb96-tapo")

    # get the most recent file from the bucket
    blobs = bucket.list_blobs(prefix="copper_cam")
    most_recent = max(blobs, key=lambda x: x.updated)
    # most_recent.download_to_filename(tmpdir + "/" + "most_recent.jpg")
    most_recent.download_to_filename("most_recent.jpg")

    message = json.loads(request.data).get('message')

    # if the message is present, do the needed
    if message is not None:
        timestamp = str(int(round(datetime.utcnow().timestamp(), 0)))
        new_blob = bucket.blob(
            "main-dataset/" +
            message + "/"
            "relabel" +
            "_" +
            timestamp
            )
        new_blob.upload_from_filename("most_recent.jpg")
        print("GCS upload complete")
        return f'Success - message: {message}'

    return 'Failure - invalid message'


request = flask.Request(environ={'REQUEST_METHOD': 'POST'})
request.data = json.dumps({"message": "class_bed"}) # type: ignore
print(main(request))
