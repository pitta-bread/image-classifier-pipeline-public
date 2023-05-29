import os


def get_first_file_path(path) -> tuple[str, str]:
    first_file_path = ""
    filename = ""
    for root, dirs, files in os.walk(path):
        if len(files) > 0:
            first_file_path = os.path.join(root, sorted(files)[0])
            filename = sorted(files)[0]
            break
    return first_file_path, filename


def get_first_subdir_path(path) -> str:
    first_subdir_path = ""
    for root, dirs, files in os.walk(path):
        if len(dirs) > 0:
            first_subdir_path = os.path.join(root, sorted(dirs)[0])
            break
    return first_subdir_path
