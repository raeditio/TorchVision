from roboflow import Roboflow
import os
from dotenv import load_dotenv

load_dotenv()
roboflow_api_key = os.getenv("roboflow_api_key")

rf = Roboflow(api_key=roboflow_api_key)
project = rf.workspace("multimeter-ocr").project("multimeter-ocr")
version = project.version(2)
dataset = version.download("yolov7")
                