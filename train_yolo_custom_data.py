
from roboflow import Roboflow
rf = Roboflow(api_key="Yek2kh0j64MOJzZtlOSM")
project = rf.workspace("joseph-nelson").project("chess-full")
version = project.version(23)
dataset = version.download("yolov11")