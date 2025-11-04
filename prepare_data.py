import os
import gdown
import argparse
import urllib.request
import zipfile
import requests

def download_covid19_model(test_session="session_09.25_00h27", model_type="RecLMIS"):
    """
    Download pretrained Covid19 model from Google Drive and save it to the correct folder structure.
    
    Args:
        test_session (str): Session name for the model
        model_type (str): Model type name
    """
    
    # Google Drive file ID extracted from the share link
    file_id = "1SEj361mYZqAZ2fYJfFquMVxjv3d6RqKm"
    
    # Create the folder structure based on the test_model.py code
    model_dir = f"./Covid19/{model_type}/{test_session}/models/"
    
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Define the output file path
    output_path = os.path.join(model_dir, f"best_model-{model_type}.pth.tar")
    
    # Download URL for Google Drive
    download_url = f"https://drive.google.com/uc?id={file_id}"
    
    print(f"Downloading model to: {output_path}")
    print(f"Creating directory structure: {model_dir}")
    
    try:
        # Download the file
        gdown.download(download_url, output_path, quiet=False)
        print(f"✅ Model downloaded successfully to: {output_path}")
        
        # Verify the file exists and has content
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"📁 File size: {file_size / (1024*1024):.2f} MB")
        else:
            print("❌ Download failed - file not found")
            
    except Exception as e:
        print(f"❌ Error downloading model: {str(e)}")
        print("💡 Make sure you have gdown installed: pip install gdown")

def download_datasets():
    """
    Download datasets folder from Google Drive.
    Uses alternative methods to handle large folders with >50 files.
    """
    
    # Google Drive folder ID extracted from the share link
    folder_id = "10q_sGJIbdggqy6HB61FSuIs5g1X7ccDc"
    
    # Create datasets directory
    datasets_dir = "./datasets/"
    os.makedirs(datasets_dir, exist_ok=True)
    
    print(f"Downloading datasets to: {datasets_dir}")
    print("This may take a while depending on the folder size...")
    
    # Method 1: Try to download as a zip file (if available)
    zip_url = f"https://drive.google.com/uc?id={folder_id}&export=download"
    zip_path = os.path.join(datasets_dir, "datasets.zip")
    
    try:
        print("🔄 Attempting to download as ZIP file...")
        
        # Try downloading as zip
        response = requests.get(zip_url, stream=True)
        if response.status_code == 200:
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the zip file
            print("📦 Extracting ZIP file...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(datasets_dir)
            
            # Remove the zip file
            os.remove(zip_path)
            print(f"✅ Datasets downloaded and extracted successfully to: {datasets_dir}")
            
        else:
            raise Exception("ZIP download failed")
            
    except Exception as e:
        print(f"❌ ZIP download method failed: {str(e)}")
        print("🔄 Trying alternative method...")
        
        # Method 2: Try gdown folder download with chunking
        try:
            print("🔄 Attempting folder download with gdown...")
            gdown.download_folder(
                f"https://drive.google.com/drive/folders/{folder_id}",
                output=datasets_dir,
                quiet=False,
                use_cookies=False,
                remaining_ok=True  # Continue even if some files fail
            )
            print(f"✅ Partial datasets downloaded to: {datasets_dir}")
            
        except Exception as e2:
            print(f"❌ Folder download also failed: {str(e2)}")
            print("🔄 Trying manual file-by-file download...")
            
            # Method 3: Manual download with known file structure
            manual_download_covid19_dataset(datasets_dir)


def manual_download_covid19_dataset(datasets_dir):
    """
    Manually download specific COVID-19 dataset files if folder download fails.
    """
    
    # Known important files for COVID-19 dataset (you may need to update these IDs)
    covid_files = {
        "Covid19_train_images.zip": "1example_file_id_1",  # Replace with actual file IDs
        "Covid19_train_masks.zip": "1example_file_id_2",   # Replace with actual file IDs
        "Covid19_test_images.zip": "1example_file_id_3",   # Replace with actual file IDs
        "Covid19_test_masks.zip": "1example_file_id_4",    # Replace with actual file IDs
    }
    
    print("📋 Downloading individual COVID-19 dataset files...")
    
    covid_dir = os.path.join(datasets_dir, "Covid19")
    os.makedirs(covid_dir, exist_ok=True)
    
    for filename, file_id in covid_files.items():
        if file_id.startswith("1example"):  # Skip placeholder IDs
            print(f"⚠️  Skipping {filename} - file ID not provided")
            continue
            
        try:
            file_path = os.path.join(covid_dir, filename)
            download_url = f"https://drive.google.com/uc?id={file_id}"
            
            print(f"📥 Downloading {filename}...")
            gdown.download(download_url, file_path, quiet=False)
            
            # If it's a zip file, extract it
            if filename.endswith('.zip'):
                print(f"📦 Extracting {filename}...")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(covid_dir)
                os.remove(file_path)  # Remove zip after extraction
                
        except Exception as e:
            print(f"❌ Failed to download {filename}: {str(e)}")
    
    print("💡 Manual download completed. Some files may be missing.")
    print("💡 If needed, manually download from:")
    print(f"💡 https://drive.google.com/drive/folders/{folder_id}")

def download_datasets_alternative():
    """
    Alternative method: Provide instructions for manual download.
    """
    folder_id = "10q_sGJIbdggqy6HB61FSuIs5g1X7ccDc"
    
    print("📋 MANUAL DOWNLOAD REQUIRED")
    print("=" * 50)
    print("The datasets folder is too large for automatic download.")
    print("Please follow these steps:")
    print()
    print("1. Open this link in your browser:")
    print(f"   https://drive.google.com/drive/folders/{folder_id}")
    print()
    print("2. Click 'Download' to download the entire folder as a ZIP")
    print("3. Extract the ZIP file to your project directory")
    print("4. Make sure the folder structure looks like:")
    print("   ./datasets/Covid19/Train/")
    print("   ./datasets/Covid19/Test/")
    print()
    print("💡 Alternative: Use Google Colab or Kaggle which have better")
    print("💡 integration with Google Drive for large downloads.")


def download_vit_model():
    """
    Download ViT-B-32.pt model from OpenAI CLIP and save it to the nets folder.
    """
    
    # Model URL from OpenAI CLIP
    model_url = "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
    
    # Use existing nets directory
    nets_dir = "./nets/"
    
    # Define the output file path
    output_path = os.path.join(nets_dir, "ViT-B-32.pt")
    
    # Check if file already exists
    if os.path.exists(output_path):
        print(f"✅ ViT-B-32.pt already exists at: {output_path}")
        file_size = os.path.getsize(output_path)
        print(f"📁 File size: {file_size / (1024*1024):.2f} MB")
        return
    
    print(f"Downloading ViT-B-32 model to: {output_path}")
    print("This may take a while depending on your internet connection...")
    
    try:
        # Download the file with progress
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded / total_size) * 100)
                print(f"\rProgress: {percent:.1f}% ({downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB)", end="")
        
        urllib.request.urlretrieve(model_url, output_path, progress_hook)
        print()  # New line after progress
        print(f"✅ ViT-B-32 model downloaded successfully to: {output_path}")
        
        # Verify the file exists and has content
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"📁 File size: {file_size / (1024*1024):.2f} MB")
        else:
            print("❌ Download failed - file not found")
            
    except Exception as e:
        print(f"❌ Error downloading ViT-B-32 model: {str(e)}")
        print("💡 Check your internet connection and try again")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download pretrained Covid19 model and datasets')
    parser.add_argument('--test_session', '-t', default='session_09.25_00h27',
                        type=str, help='Session name for the model')
    parser.add_argument('--model_type', '-m', default='RecLMIS',
                        type=str, help='Model type name')
    parser.add_argument('--download_model', action='store_true', 
                        help='Download the pretrained model')
    parser.add_argument('--download_datasets', action='store_true',
                        help='Download the datasets folder')
    parser.add_argument('--download_vit', action='store_true',
                        help='Download the ViT-B-32 model')
    parser.add_argument('--download_all', action='store_true',
                        help='Download model, datasets, and ViT-B-32')
    parser.add_argument('--download_datasets_manual', action='store_true',
                        help='Show manual download instructions for datasets')
    
    args = parser.parse_args()
    
    # Install gdown if not available
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        os.system("pip install gdown")
        import gdown
    
    # Download based on arguments
    if args.download_all:
        print("🚀 Downloading model, datasets, and ViT-B-32...")
        download_covid19_model(args.test_session, args.model_type)
        print("\n" + "="*50 + "\n")
        download_datasets()
        print("\n" + "="*50 + "\n")
        download_vit_model()
    elif args.download_model:
        print("🚀 Downloading model only...")
        download_covid19_model(args.test_session, args.model_type)
    elif args.download_datasets:
        print("🚀 Downloading datasets only...")
        download_datasets()
    elif args.download_datasets_manual:
        download_datasets_alternative()
    elif args.download_vit:
        print("🚀 Downloading ViT-B-32 model only...")
        download_vit_model()
    else:
        # Default behavior - download both
        print("🚀 No specific option selected. Downloading both model and datasets...")
        download_covid19_model(args.test_session, args.model_type)
        print("\n" + "="*50 + "\n")
        download_datasets()