import os
import requests
import zipfile
from tqdm import tqdm

def download_glove(dest_folder):
    """
    Downloads GloVe embeddings (glove.6B.zip) and extracts them.
    """
    url = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
    zip_path = os.path.join(dest_folder, "glove.6B.zip")
    
    # Check if extracted file already exists (e.g. 100d)
    target_file = os.path.join(dest_folder, "glove.6B.100d.txt")
    if os.path.exists(target_file):
        print(f"GloVe embeddings found at {target_file}")
        return target_file

    print(f"Downloading GloVe embeddings from {url}...")
    
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(zip_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
        
    print("Done!")
    return target_file

if __name__ == "__main__":
    # Download to data folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    download_glove(data_dir)
