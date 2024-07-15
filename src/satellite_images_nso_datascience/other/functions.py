from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
import requests


# function to upload the annotations to the cloud, for searching with other parties
def upload_to_blob(apath, connection_string, directory_blob):

    container_name = "satellite-images-nso"
    blob_name = directory_blob + os.path.basename(apath)

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)

    with open(apath, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)


def download_file(url, local_path):

    # Send a GET request to the URL to download the zip file
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Write the content of the response (the zip file) to a local file
        with open(local_path, "wb") as file:
            file.write(response.content)

        print(f"file has been downloaded to {local_path}")
    else:
        print("Failed to download file:", response.status_code)
