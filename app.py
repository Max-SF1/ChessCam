"""
this is something called a docstring, pylint really likes those. 
they enforce good coding practices. The more you know! 
app.py - Runs object detection on chess images using Ultralytics YOLO and displays results.
"""
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Run inference on 'bus.jpg' with arguments
res = model.predict(
            "/workspace/image/view-chess-pieces-with-shatter-effect.jpg",
            save=True,
            imgsz=640,
            conf=0.5,
            )

plt.imshow(res[0].plot())
plt.axis("off")
plt.tight_layout()
plt.title("First Detection")
plt.show(block=True)



np.zeros_like(res[0].show())
