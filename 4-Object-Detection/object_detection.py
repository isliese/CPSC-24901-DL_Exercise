# Swayam Shree

import torch

def main():
    #---------------------------------------------------------------------
    # 1) Define the actual objects that you *know* are in the image
    #    (This is just for demonstration; you'll fill in the items you expect.)
    #---------------------------------------------------------------------
    actual_objects = ["bottle", "wallet", "plant"]  # Example ground-truth objects in your image

    # Print the objects we know are in the image
    print("Actual objects in the image:", actual_objects)

    #---------------------------------------------------------------------
    # 2) Load a pretrained YOLOv5 model via Torch Hub.
    #    We use 'yolov5s' (small model) for example. 
    #    The first time this runs, it will download the model weights.
    #---------------------------------------------------------------------
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    #---------------------------------------------------------------------
    # 3) Perform inference on your local image.
    #---------------------------------------------------------------------
    image_path = "objects.jpg"  # Update if your image is stored elsewhere
    results = model(image_path)

    #---------------------------------------------------------------------
    # 4) Extract predictions and print them.
    #    The 'results.pandas().xyxy[0]' call returns a Pandas DataFrame
    #    with columns: [xmin, ymin, xmax, ymax, confidence, class, name].
    #    - (xmin, ymin) is the top-left corner
    #    - (xmax, ymax) is the bottom-right corner
    #---------------------------------------------------------------------
    predictions = results.pandas().xyxy[0]

    # Collect unique predicted objects:
    predicted_objects = predictions["name"].unique().tolist()
    print("Predicted objects:", predicted_objects)

    # Print bounding boxes for each detected object
    print("\nDetailed detections (object name and bounding box):")
    for i in range(len(predictions)):
        name = predictions.loc[i, "name"]
        xmin = predictions.loc[i, "xmin"]
        ymin = predictions.loc[i, "ymin"]
        xmax = predictions.loc[i, "xmax"]
        ymax = predictions.loc[i, "ymax"]

        print(f" - {name} at [{xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f}]")

if __name__ == "__main__":
    main()
