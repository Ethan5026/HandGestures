import requests
import zipfile
import io
from boxsdk import OAuth2, Client
from concurrent.futures import ThreadPoolExecutor

# Box authentication (replace with your credentials)
def authenticate_box():
    auth = OAuth2(
        client_id='cseg4cofbhc3qefojxeporq620htm7t1',
        client_secret='f97d0oJyDHCznl8LCyHtcYRynlRuez6T',
        access_token='F1NNoG5bplyANGHFGKK6qC1b8JaJm7pw',
    )
    return Client(auth)

# Upload a single file to Box
def upload_file_to_box(box_client, folder_id, file_name, file_content):
    folder = box_client.folder(folder_id)
    total_size = len(file_content)
    if total_size > 20*1024*1024:  # Use multipart for files >20MB
        upload_session = folder.create_upload_session(total_size, file_name)
        part_size = 20*1024*1024  # 20MB chunks
        parts = []
        for i in range(0, total_size, part_size):
            chunk = file_content[i:i + part_size]
            part = upload_session.upload_part_bytes(chunk, i, total_size)
            parts.append(part)
        upload_session.commit(total_file_size=total_size, parts=parts)
    else:
        folder.upload_stream(io.BytesIO(file_content), file_name)
    print(f"Uploaded {file_name}")

# Stream, unzip, and upload contents
def upload_unzipped_from_url(box_client, download_url, folder_id, chunk_size=10*1024*1024):  # 10MB chunks
    # Stream download from URL
    response = requests.get(download_url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    print(f"Starting download of ZIP ({total_size / (1024*1024):.2f} MB)")

    # Buffer the ZIP in memory
    zip_buffer = io.BytesIO()
    bytes_downloaded = 0
    for chunk in response.iter_content(chunk_size=chunk_size):
        if chunk:
            zip_buffer.write(chunk)
            bytes_downloaded += len(chunk)
            print(f"Downloaded {bytes_downloaded / (1024*1024):.2f} MB", end='\r')

    print("\nExtracting and uploading contents...")
    zip_buffer.seek(0)

    # Open ZIP and extract files
    with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
        # Use ThreadPoolExecutor to upload files in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for file_info in zip_ref.infolist():
                if not file_info.is_dir():  # Skip directories
                    file_name = file_info.filename
                    file_content = zip_ref.read(file_name)
                    futures.append(
                        executor.submit(upload_file_to_box, box_client, folder_id, file_name, file_content)
                    )
            # Wait for all uploads to complete
            for future in futures:
                future.result()

    print("All files uploaded successfully.")

def main():

    dislike_url = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset/dislike.zip"
    dislike_folder = "309095126275"

    like_url = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset/like.zip"
    like_folder = "309094821207"

    ok_url="https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset/ok.zip"
    ok_folder = "309093673935"

    palm_url = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset/palm.zip"
    palm_folder = "309095378224"

    peace_url = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset/peace.zip"
    peace_folder = "309093926036"

    downloads = [
        [dislike_url, dislike_folder],
        [like_url, like_folder],
        [ok_url, ok_folder],
        [palm_url, palm_folder],
        [peace_url, peace_folder]
        ]
    # Authenticate with Box
    box_client = authenticate_box()
    for download in downloads:
        # Upload the file
        upload_unzipped_from_url(box_client, download[0], download[1])

if __name__ == "__main__":
    main()



