from ultralytics import YOLO
import torch

def main():
    
    model = YOLO('Kitti\\weight_150_epoch\\best.pt')  # Load model

    # Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model

main()