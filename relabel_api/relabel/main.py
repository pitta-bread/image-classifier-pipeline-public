import functions_framework
import tempfile
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
    tmpdir = tempfile.gettempdir()

    # create google cloud storage client
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("home-assistant-pb96-tapo")

    # get the most recent file from the bucket
    blobs = bucket.list_blobs(prefix="copper_cam")
    most_recent = max(blobs, key=lambda x: x.updated)
    most_recent.download_to_filename("most_recent.jpg")

    message = request.get_json().get('message', None)

    # if the message is present, 
    if message is not None:
        

    return 'Failure - invalid message'


print(main(None))
