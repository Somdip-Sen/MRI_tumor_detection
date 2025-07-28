# python download_latest_checkpoint.py 172.16.82.56 pguha4 --remote_path '~/Home/importedFile/MRI_Tumour/Checkpoints/training_checkpoint.pth'
import argparse
import getpass
import os
import sys
import paramiko
from scp import SCPClient

import RenameTheLatestCheckpoint


def download_checkpoint(hostname, username, remote_path, local_path, password=None):
    """
    Connects to a remote server via SSH and downloads a file using SCP.
    """
    client = None
    try:
        # Get password securely without showing it on the screen
        password = password if password else getpass.getpass(f"Enter password for {username}@{hostname}: ")
        # Establish SSH connection
        print(f"Connecting to {hostname}...")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname, username=username, password=password, port=22)
        print("Connection successful.")

        # Use the SCP client to download the file
        with SCPClient(client.get_transport()) as scp_client:
            print(f"Downloading remote file: {remote_path}")
            print(f"Saving to local path: {os.path.abspath(local_path)}")
            scp_client.get(remote_path, local_path, recursive=True)

        print("\n✅ File downloaded successfully!")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}", file=sys.stderr)
    finally:
        if client:
            client.close()
            print("Connection closed.")


if __name__ == "__main__":
    RenameTheLatestCheckpoint.archive_latest_run()  #
    parser = argparse.ArgumentParser(
        description="Download a file from a remote server using SCP."
    )
    parser.add_argument("hostname", help="The remote server's hostname or IP address.")
    parser.add_argument("username", help="Your username on the remote server.")
    parser.add_argument(
        "--remote_path",
        required=True,
        help="The full path to the file on the remote server."
    )
    parser.add_argument(
        "--local_path",
        default="./Checkpoints",
        help="The local directory to save the file in. Defaults to the current directory."
    )

    args = parser.parse_args()

    download_checkpoint(
        hostname=args.hostname,
        username=args.username,
        password="NineLabs@4",
        remote_path=args.remote_path,
        local_path=args.local_path
    )
