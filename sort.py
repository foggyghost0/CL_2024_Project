import os
import shutil

def move_images_and_cleanup(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                shutil.move(file_path, directory)
        for name in dirs:
            dir_path = os.path.join(root, name)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)

if __name__ == "__main__":
    directory = "images1024x1024"
    move_images_and_cleanup(directory)