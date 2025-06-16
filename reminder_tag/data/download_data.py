#!/usr/bin/env python3

import os
import sys
import requests
import json
import zipfile
import argparse
from urllib.parse import urlparse
from pathlib import Path
from tqdm import tqdm


class DatasetDownloader:
    """A class for downloading and extracting Visual Dialogue and COCO datasets."""
    
    # Known URLs for VisDial dataset files (updated with actual accessible URLs)
    VISDIAL_URLS = {
        "train": "https://computing.ece.vt.edu/~abhshkdz/data/visdial/visdial_1.0_train.zip",
        "val": "https://computing.ece.vt.edu/~abhshkdz/data/visdial/visdial_1.0_val.zip",
        "test": "https://computing.ece.vt.edu/~abhshkdz/data/visdial/visdial_1.0_test.zip",
        "val_images": "https://computing.ece.vt.edu/~abhshkdz/data/visdial/VisualDialog_val2018.zip"
    }
    
    # Known URLs for COCO dataset files
    COCO_URLS = {
        "train2014": "http://images.cocodataset.org/zips/train2014.zip",
        "val2014": "http://images.cocodataset.org/zips/val2014.zip",
        "val2018": "http://images.cocodataset.org/zips/val2018.zip",
        "annotations_2014": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
        "instances_train2014": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    }
    
    def __init__(self, data_dir="./data"):
        """
        Initialize the dataset downloader.
        
        Args:
            data_dir (str): Root directory to store the datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url, destination, chunk_size=8192):
        """
        Download a file from a URL to a destination path with progress bar.
        
        Args:
            url (str): URL to download from
            destination (Path): Local path to save the file
            chunk_size (int): Size of chunks to download at a time
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Downloading from: {url}")
            print(f"Saving to: {destination}")
            
            # Create directory if it doesn't exist
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if file already exists
            if destination.exists():
                print(f"File already exists: {destination}")
                return True
            
            # Start the download
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get the total file size if available
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(destination, 'wb') as f:
                with tqdm(
                    total=total_size, 
                    unit='B', 
                    unit_scale=True, 
                    desc=destination.name
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"Successfully downloaded: {destination}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error downloading {url}: {e}")
            return False
    
    def extract_zip(self, zip_path, extract_to):
        """
        Extract a zip file to a destination folder.
        
        Args:
            zip_path (Path): Path to the zip file
            extract_to (Path): Directory to extract to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            extract_to.mkdir(parents=True, exist_ok=True)
            
            print(f"Extracting {zip_path.name} to {extract_to}")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get total size for progress bar
                file_list = zip_ref.infolist()
                total_size = sum(file.file_size for file in file_list)
                
                # Extract with progress bar
                with tqdm(
                    total=total_size, 
                    unit='B', 
                    unit_scale=True, 
                    desc=f"Extracting {zip_path.name}"
                ) as pbar:
                    for file in file_list:
                        zip_ref.extract(file, extract_to)
                        pbar.update(file.file_size)
            
            print(f"Successfully extracted: {zip_path.name}")
            return True
            
        except Exception as e:
            print(f"Error extracting {zip_path}: {e}")
            return False
    
    def download_visdial(self, splits=None):
        """
        Download and extract VisDial dataset files.
        
        Args:
            splits (list): List of splits to download, can include 'train', 'val', 'test'
        """
        if not splits:
            splits = ['train', 'val', 'test']
        
        visdial_dir = self.data_dir / 'visdial'
        visdial_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"=== Downloading VisDial dataset files to {visdial_dir} ===")
        
        success_count = 0
        total_count = len(splits)
        
        # Download and extract each split
        for split in splits:
            if split not in self.VISDIAL_URLS:
                print(f"Warning: Unknown split '{split}', skipping.")
                continue
            
            zip_path = visdial_dir / f"visdial_1.0_{split}.zip"
            
            # Download the zip file
            url = self.VISDIAL_URLS[split]
            print(f"\n--- Downloading {split} split ---")
            
            if self.download_file(url, zip_path):
                # Extract the zip file
                extract_dir = visdial_dir / 'data'
                if self.extract_zip(zip_path, extract_dir):
                    success_count += 1
        
        # Download validation images if val split is requested
        if 'val' in splits:
            print(f"\n--- Downloading validation images ---")
            val_images_zip = visdial_dir / "VisualDialog_val2018.zip"
            
            if self.download_file(self.VISDIAL_URLS["val_images"], val_images_zip):
                # Extract validation images
                images_dir = visdial_dir / 'images'
                self.extract_zip(val_images_zip, images_dir)
        
        print(f"\nVisDial download completed: {success_count}/{total_count} splits successful")
    
    def download_coco(self, datasets=None):
        """
        Download and extract COCO dataset files.
        
        Args:
            datasets (list): List of COCO datasets to download
        """
        if not datasets:
            datasets = ['train2014', 'val2014', 'annotations_2014']
        
        coco_dir = self.data_dir / 'coco'
        coco_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"=== Downloading COCO dataset files to {coco_dir} ===")
        
        success_count = 0
        total_count = len(datasets)
        
        # Download and extract each dataset
        for dataset in datasets:
            if dataset not in self.COCO_URLS:
                print(f"Warning: Unknown COCO dataset '{dataset}', skipping.")
                continue
            
            zip_path = coco_dir / f"{dataset}.zip"
            
            # Download the zip file
            url = self.COCO_URLS[dataset]
            print(f"\n--- Downloading {dataset} ---")
            
            if self.download_file(url, zip_path):
                # Extract the zip file
                if self.extract_zip(zip_path, coco_dir):
                    success_count += 1
        
        print(f"\nCOCO download completed: {success_count}/{total_count} datasets successful")
    
    def setup_directory_structure(self):
        """
        Set up the directory structure for the project.
        """
        print("=== Setting up directory structure ===")
        
        # Define the directory structure
        dirs = [
            self.data_dir / 'visdial' / 'data',
            self.data_dir / 'visdial' / 'images' / 'val2018',
            self.data_dir / 'coco' / 'train2014',
            self.data_dir / 'coco' / 'val2014',
            self.data_dir / 'coco' / 'annotations',
            self.data_dir / 'outputs',
            self.data_dir / 'models'
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {dir_path}")
        
        # Create a README file in the data directory
        readme_content = """# Visual Dialogue Dataset Directory

This directory contains the downloaded VisDial and COCO datasets.

## Directory Structure:
- visdial/
  - data/               # VisDial JSON files (train, val, test)
  - images/             # VisDial validation images
- coco/
  - train2014/          # COCO training images
  - val2014/            # COCO validation images
  - annotations/        # COCO annotation files
- outputs/              # Model outputs and results
- models/               # Saved model checkpoints

## Files:
- visdial_1.0_train.json: Training dialogues
- visdial_1.0_val.json: Validation dialogues
- visdial_1.0_test.json: Test dialogues
- instances_train2014.json: COCO training annotations
- instances_val2014.json: COCO validation annotations
"""
        
        readme_path = self.data_dir / 'README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"✓ Created README file: {readme_path}")
    
    def verify_downloads(self):
        """
        Verify that the essential files have been downloaded correctly.
        """
        print("\n=== Verifying downloads ===")
        
        essential_files = [
            self.data_dir / 'visdial' / 'data' / 'visdial_1.0_train.json',
            self.data_dir / 'visdial' / 'data' / 'visdial_1.0_val.json',
        ]
        
        verification_passed = True
        
        for file_path in essential_files:
            if file_path.exists():
                file_size = file_path.stat().st_size
                print(f"✓ {file_path.name}: {file_size:,} bytes")
            else:
                print(f"✗ Missing: {file_path}")
                verification_passed = False
        
        if verification_passed:
            print("✓ All essential files verified successfully!")
        else:
            print("✗ Some files are missing. Please check the download process.")
        
        return verification_passed


def main():
    """Main function to parse arguments and execute downloads."""
    parser = argparse.ArgumentParser(
        description='Download VisDial and COCO datasets for Visual Dialogue training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_data.py                                # Download all datasets
  python download_data.py --visdial_only                 # Download only VisDial
  python download_data.py --coco_only                    # Download only COCO
  python download_data.py --data_dir ./my_data           # Custom data directory
  python download_data.py --visdial_splits train val     # Specific VisDial splits
        """
    )
    
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='./data', 
        help='Root directory to store the datasets (default: ./data)'
    )
    parser.add_argument(
        '--visdial_only', 
        action='store_true', 
        help='Download only VisDial dataset'
    )
    parser.add_argument(
        '--coco_only', 
        action='store_true', 
        help='Download only COCO dataset'
    )
    parser.add_argument(
        '--split', 
        nargs='+', 
        default=['train', 'val', 'test'], 
        help='VisDial splits to download (choices: train, val, test)'
    )
    parser.add_argument(
        '--coco_datasets', 
        nargs='+', 
        default=['train2014', 'val2014', 'annotations_2014'], 
        help='COCO datasets to download (choices: train2014, val2014, val2018, annotations_2014)'
    )
    parser.add_argument(
        '--skip_verification', 
        action='store_true', 
        help='Skip verification of downloaded files'
    )
    
    args = parser.parse_args()
    
    print("🚀 Visual Dialogue Dataset Download Script")
    print("=" * 50)
    
    # Initialize the downloader
    downloader = DatasetDownloader(args.data_dir)
    
    # Setup the directory structure
    downloader.setup_directory_structure()
    
    # Download datasets based on flags
    if args.visdial_only:
        downloader.download_visdial(args.visdial_splits)
    elif args.coco_only:
        downloader.download_coco(args.coco_datasets)
    else:
        # Download both datasets by default
        downloader.download_visdial(args.split)
        downloader.download_coco(args.coco_datasets)
    
    # Verify downloads unless skipped
    if not args.skip_verification:
        downloader.verify_downloads()
    
    print("\n🎉 Download process completed!")
    print(f"📁 Data stored in: {downloader.data_dir.absolute()}")
    print("\nNext steps:")
    print("1. Check the data directory for downloaded files")
    print("2. Run training scripts with the downloaded data")
    print("3. Refer to the project README for usage instructions")


if __name__ == "__main__":
    main()