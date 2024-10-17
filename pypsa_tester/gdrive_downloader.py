from google_drive_downloader import GoogleDriveDownloader as gdd
import re
import os
import argparse

def download_and_unzip_gdrive(file_id, file_path, enable_progress, enable_log):
    """

    Function to download data from google drive

    Inputs
    ------
    file_id : str
        gdrive id to download
    file_path : str
        path of the file to save
    enable_progress : Bool
        When true the progress bar to download data is enabled
    enable_log : Bool
        When true log messages are enabled   
    
    Outputs
    -------
    True when download is successful, False otherwise
    """
    # remove file if it exists
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # download file from google drive
    gdd.download_file_from_google_drive(
        file_id=file_id,
        dest_path=file_path,
        showsize=enable_progress,
        unzip=False,
        log=enable_log,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='gdrive downloader')
    parser.add_argument(
        'url',
        type=str,
        help='url of the file to download'
    )
    parser.add_argument(
        'file_path',
        type=str,
        help='path of the file to save'
    )
    parser.add_argument(
        '--enable_progress',
        action='store_true',
        help='When true the progress bar to download data is enabled',
        default=False,
    )
    parser.add_argument(
        '--enable_log',
        action='store_true',
        help='When true the progress bar to download data is enabled',
        default=False,
    )

    args = parser.parse_args()

    download_and_unzip_gdrive(args.url, args.file_path, args.enable_progress, args.enable_log)