"""
this is something called a docstring, pylint really likes those. 
they enforce good coding practices. The more you know! 
app.py - Runs object detection on chess images using Ultralytics YOLO and displays results.
pylint also likes the last line to be empty 
- i guess it means the file ended deliberately and the programmer didn't get shot in the middle of wri 
"""
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


print("yo")

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")
print("first checkpoint")

# Run inference on 'bus.jpg' with arguments
res = model.predict(
            "/workspace/image/view-chess-pieces-with-shatter-effect.jpg",
            save=False, #WAS TRUE
            imgsz=640,
            conf=0.5,
            )

print("second checkpoint")
plt.imshow(res[0].plot())
plt.axis("off")
plt.tight_layout()
plt.title("First Detection")
plt.show(block=True)



np.zeros_like(res[0].show())
print("done")
