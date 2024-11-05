import os
import requests
import zipfile
import threading

# URLs for the COCO 2017 dataset
urls = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

# Define the directory structure
base_dir = "data"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
annotations_dir = os.path.join(base_dir, "annotations")

# Ensure the folder structure exists
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

# Function to download and extract zip files
def download_and_extract(url, output_dir):
    zip_path = os.path.join(output_dir, os.path.basename(url))
    try:
        # Download the file
        print(f"Starting download for {url}...")
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses

        with open(zip_path, "wb") as f:
            total_length = response.headers.get('content-length')
            if total_length is None:  # No content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        dl += len(chunk)
                        f.write(chunk)
                        done = int(50 * dl / total_length)
                        print(f"\r[{'=' * done}{' ' * (50 - done)}] {dl / total_length:.2%}", end='')
        print(f"\nDownloaded {zip_path}")

        # Extract the zip file
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extracted to {output_dir}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}. Error: {e}")
    except zipfile.BadZipFile:
        print(f"Failed to unzip {zip_path}. The file may be corrupted.")

# Main function to start multithreaded downloads
def main():
    threads = []
    
    # Create a thread for each download task
    threads.append(threading.Thread(target=download_and_extract, args=(urls["train_images"], train_dir)))
    threads.append(threading.Thread(target=download_and_extract, args=(urls["val_images"], val_dir)))
    threads.append(threading.Thread(target=download_and_extract, args=(urls["annotations"], annotations_dir)))
    
    # Start each thread
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()