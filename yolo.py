import os
import urllib.request

def download_model_files():
    # YOLOv3 model files URLs
    model_urls = {
        'yolov3.weights': 'https://pjreddie.com/media/files/yolov3.weights',
        'yolov3.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
        'coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
    }

    print("Downloading YOLOv3 model files...")
    for file_name, url in model_urls.items():
        file_path = os.path.join(os.getcwd(), file_name)
        urllib.request.urlretrieve(url, file_path)
        print(f"Downloaded {file_name}")

    print("Download completed!")

if __name__ == "__main__":
    download_model_files()
