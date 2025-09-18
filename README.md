
```
   ________                   ______                __
  / ____/ /_  ___  __________/ ____/___ _____ ___  / /
 / /   / __ \/ _ \/ ___/ ___/ /   / __ `/ __ `__ \/ /
/ /___/ / / /  __(__  |__  ) /___/ /_/ / / / / / /_/
\____/_/ /_/\___/____/____/\____/\__,_/_/ /_/ /_(_)
```
***

# ChessCam! using YOLO11 on a physical game to generate realtime move suggestions! 

## [Watch the demo on YouTube](https://www.youtube.com/watch?v=ubye3jiIMzM) 

Additionally, you can see our system handling [**Captures, and Illegal Moves**](https://www.youtube.com/watch?v=eKCE8y1vlIw&feature=youtu.be), complex moves like **[En-Passant](https://www.youtube.com/watch?v=2sT8WWDWMfo)**, and **[Error Recovery](https://www.youtube.com/watch?v=qV9FdVCifS0)**! 

You can read more about the development process in **ChessCam_Project_Report.pdf**, or see our **poster**.

(Full disclosure: our system does not handle moves like castling or promotion, since the project had limited scope). 

### Overview 

<p align="center">
  <img src="https://github.com/user-attachments/assets/00e8d395-7ecd-48d3-86ec-980bfdfd2ef5" alt="Twitter Bot Image" width="600"/>
</p>

* An intel realsense D435 camera's color stream is accessed using pyrealsense
* Every frame, Ultralytics' YOLOv11s provides piece classifications
* We track the pieces using a SORT (simple online realtime tracking) algorithm we implemented, and associate the new detections to tracked piece objects.
* Using OpenCV's functions we localize the pieces to their appropriate square using a Homography.
* We assemble a Python-Chess board object with the current game state, perform legality checks, and query stockfish for a move recommendation.

### Training: 
Training was done in Roboflow using [The Following Dataset](https://app.roboflow.com/bonus-bghtl/chess-newest-71ioo/2). We did not allocate images to a test set considering the system's performance itself is the test. About 20% of the images were used for validation. The dataset is comprised off of a mix of Roboflow's dataset, and one we made on our own using the `recorder.py` script that generates images from the realsense camera stream. We made sure to add both null images, and images with occlusions, since those situations are common in our use case (people move pieces and their hands occlude them frequently). 

### Localization:
A homography is a linear transformation that maps between two planes, and we can compute it with matching points. Once we have a homography, we use the tracked pieces' bounding box parameters to compute a point that represents accurately the square the pieces are on. Here's how we do it: 

<p align="center">
  <img src="https://github.com/user-attachments/assets/05e5c7e8-a762-4a3f-890a-d09b524d3911" alt="Twitter Bot Image" width="300"/>
</p>

And you can **[Watch the Following Animation](https://youtu.be/ZMWNphRdz5o)** to see how we compute the Homography using OpenCV's `findChessBoardCorners`, (we then sort them to row major to map to a planar board's corners), and use the computed homography to map pieces to their correct squares. 

### Major File Overview: 
You can read more about the development process in **ChessCam_Project_Report.pdf**, or see our **poster**.
| File                          | Description                                                                                           |
|-------------------------------|-------------------------------------------------------------------------------------------------------|
| **src/scripts/online_inference.py**                | Main file for preforming the inference of chess position, and tracking |
| **src/scripts/recorder.py** | A file for extracting images from the camera using pyrealsense, used to generate training images   |
| **src/scripts/train_yolo11_object_detection_on_custom_dataset.ipynb**   | modified Ultralytics' notebook for training the YOLOv11s model|
| **src/scripts/utils/supporting_structs.py** |  Supporting structures for object tracking, such as PieceManager, and Associator.                   |
| **src/scripts/utils/chess_supporting_structs.py** | Supporting structures for chess logic, interfacing with stockfish and move inference.                             |
|**src/Dockerfile** and **src/Docker-compose.yaml**| set up the image and container for the development environment|

## Setup

We want to work in an environment that has both git, github, VSCode, and Docker.
To do that, we first create a github account, download a VScode git extension, download docker, and a
VScode extension for docker too.

- we select an isolated folder where a localized instance of our project takes place.
- Using `ctrl+shift+p` and then selecting the command `Docker: Add Docker files to Workspace` we get a few empty docker files
- we need to give the container permission to access the camera, this is done with the command `xhost +local:docker`
- make sure you have stockfish installed and that the adress in `chess_supporting_structs` matches its location on your machine.
---
