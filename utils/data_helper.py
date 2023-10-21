from azure.storage.blob import BlobServiceClient
import tempfile
import os
from vyper import v

v.automatic_env()
service = BlobServiceClient(account_url=v.get("BLOB_ACCOUNT"), credential=v.get("BLOB_KEY"))



def load_blobs_from_cotainer(path="wav"):
    path = os.sep.join(path.split('/'))
    container_name = path.split(os.sep)[0]
    folder_name = '/'.join(path.split(os.sep)[1:])

    container_client_input = service.get_container_client(container_name)
    handeled_files = 0
    for blob in container_client_input.list_blobs(name_starts_with=folder_name):
        handeled_files += 1
        if handeled_files % 100 == 0:
            print(f"Handeling file {handeled_files}")
        yield blob.name, container_client_input.get_blob_client(blob.name).download_blob()

def list_blobs(container_name, sub_folder = None):
    container_client_input = service.get_container_client(container_name)
    if sub_folder:
        return container_client_input.list_blobs(name_starts_with=f"{sub_folder}/")
    else:
        return container_client_input.list_blobs()
    
def download_blob(container_name, blob_name):
    container_client_input = service.get_container_client(container_name)
    return container_client_input.get_blob_client(blob_name).download_blob()

def DataUploader(data, output_path, parent_path="output"):
    path = os.sep.join([*parent_path.split('/'), *output_path.replace('\\', '/').split('/')])
    container_name = path.split(os.sep)[0]
    output_path = os.sep.join(path.split(os.sep)[1:])
    print(f'SAVING to Blob {container_name}, at directory {output_path}')
    blob_client = service.get_blob_client(container_name, output_path)
    blob_client.upload_blob(data, blob_type="BlockBlob", overwrite=True)

def is_exists(container_name, path):
    container_client = service.get_container_client(container_name)
    return container_client.get_blob_client(path).exists()


def get_tmp_file(audio_data, suffix=None):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        audio_data.readinto(temp_file)
    return temp_file.name