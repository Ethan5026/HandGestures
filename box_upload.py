import os
import sys
import requests
import zipfile
import io
from boxsdk import OAuth2, Client
from concurrent.futures import ThreadPoolExecutor
import tempfile


def authenticate_box():
    auth = OAuth2(
        client_id=os.getenv('BOX_CLIENT_ID'),
        client_secret=os.getenv('BOX_CLIENT_SECRET'),
        access_token=os.getenv('BOX_ACCESS_TOKEN'),
    )
    return Client(auth)


def upload_file_to_box(box_client, folder_id, file_name, file_content):
    folder = box_client.folder(folder_id)
    folder.upload_stream(io.BytesIO(file_content), file_name)
    print(f"Uploaded {file_name}")


def process_zip_chunk(box_client, folder_id, temp_file_path, offset):
    try:
        with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                for file_info in zip_ref.infolist():
                    if not file_info.is_dir():
                        try:
                            file_content = zip_ref.read(file_info.filename)
                            futures.append(
                                executor.submit(upload_file_to_box, box_client, folder_id, file_info.filename,
                                                file_content)
                            )
                        except (zipfile.BadZipFile, KeyError):
                            continue  # Skip incomplete files
                for future in futures:
                    future.result()
    except zipfile.BadZipFile:
        pass  # Chunk might be incomplete; move to next


def upload_unzipped_from_url(box_client, download_url, folder_id, chunk_size=20 * 1024 * 1024):
    response = requests.head(download_url)
    total_size = int(response.headers.get('content-length', 0))
    print(f"Total size: {total_size / (1024 * 1024):.2f} MB")

    offset = 0
    while offset < total_size:
        end_byte = min(offset + chunk_size - 1, total_size - 1)
        headers = {'Range': f'bytes={offset}-{end_byte}'}

        with requests.get(download_url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MB sub-chunks
                    temp_file.write(chunk)
                temp_file_path = temp_file.name

        print(f"Processing chunk {offset / (1024 * 1024):.2f} - {end_byte / (1024 * 1024):.2f} MB")
        process_zip_chunk(box_client, folder_id, temp_file_path, offset)
        os.unlink(temp_file_path)  # Delete after processing
        offset += chunk_size

    print("All chunks processed.")


def main(gestures):
    downloads = {
        "dislike": [
            "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset/dislike.zip",
            "309095126275"],
        # Add other gestures as needed
    }
    box_client = authenticate_box()
    for gesture in gestures:
        if gesture in downloads:
            print(f"Starting {gesture}")
            upload_unzipped_from_url(box_client, downloads[gesture][0], downloads[gesture][1])
        else:
            print(f"Gesture not found: {gesture}")


if __name__ == "__main__":
    gestures = sys.argv[1:] if len(sys.argv) > 1 else []
    if gestures:
        main(gestures)
    else:
        print("No arguments provided.")