import os

from hipscat.io.file_io.file_io import get_fs

def copy_tree_fs_to_fs(
        fs1_source: str, fs2_destination: str,
        storage_options1: dict = None, storage_options2: dict = None,
        verbose=False
    ):
    """Recursive Copies directory from one filesystem to the other.

    Args:
        fs1_source: location of source directory to copy
        fs2_destination: location of destination directory to for fs1 to be written two
        storage_options1: dictionary that contains abstract filesystem1 credentials
        storage_options2: dictionary that contains abstract filesystem2 credentials
    """

    source_fs, source_fp = get_fs(fs1_source, storage_options=storage_options1)
    destination_fs, desintation_fp = get_fs(fs2_destination, storage_options=storage_options2)
    copy_dir(source_fs, source_fp, destination_fs, desintation_fp, verbose=verbose)


def copy_dir(source_fs, source_fp, destination_fs, desintation_fp, verbose=False, chunksize=1024*1024):
    """Recursive method to copy directories and their contents.

    Args:
        fs1: fsspec.filesystem for the source directory contents
        fs1_pointer: source directory to copy content files
        fs2: fsspec.filesytem for destination directory
        fs2_pointer: destination directory for copied contents
    """
    destination_folder = os.path.join(desintation_fp, source_fp.split("/")[-1])
    if destination_folder[-1] != "/":
        destination_folder += "/"
    if not destination_fs.exists(destination_folder):
        if verbose:
            print(f"Creating destination folder: {destination_folder}")
        destination_fs.makedirs(destination_folder, exist_ok=True)

    dir_contents = source_fs.listdir(source_fp)
    files = [x for x in source_fs.listdir(source_fp) if x["type"] == "file"]

    for _file in files:
        destination_fname = os.path.join(destination_folder, _file["name"].split("/")[-1])
        if verbose:
            print(f'Copying file {_file["name"]} to {destination_fname}')
        with source_fs.open(_file["name"], "rb") as source_file:
            with destination_fs.open(destination_fname, "wb") as destination_file:
                while True:
                    chunk = source_file.read(chunksize)
                    if not chunk:
                        break
                    destination_file.write(chunk)

    dirs = [x for x in dir_contents if x["type"] == "directory"]
    for _dir in dirs:
        copy_dir(source_fs, _dir["name"], destination_fs, destination_folder, chunksize=chunksize, verbose=verbose)

if __name__ == "__main__":
    
    source_pw = f"{os.getcwd()}/../tests/data"
    target_pw = "new_protocol:///path/to/pytest/lsdb"

    target_so = {
        "valid_storage_option_param1" : os.environ.get("NEW_PROTOCOL_PARAM1"),
        "valid_storage_option_param1" : os.environ.get("NEW_PROTOCOL_PARAM2"),
    }
    copy_tree_fs_to_fs(
        source_pw, target_pw, {}, target_so, verbose=True
    )