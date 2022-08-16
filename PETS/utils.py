import os

def confirm_path(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)