"""a first test Harel wrote."""
from ultralytics import YOLOE

# Initialize a YOLOE model
model = YOLOE("yoloe-11s-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes

# Set text prompt to detect person and bus. You only need to do this once after you load the model.
names = ["chess piece", "pawn"]
model.set_classes(names, model.get_text_pe(names))

# Run detection on the given image
results = model.predict("/workspace/runs/detect/predict2/view-chess-pieces-with-shatter-effect.jpg",
                        save=True)

# Show results
results[0].show()