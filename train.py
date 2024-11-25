from ultralytics import YOLO
import torch


def main():
    model = YOLO('Kitti/yolo11s.pt')  # Load model

    train = model.train(data='Kitti\kitti.yaml',
                        epochs=150,
                        imgsz=640,
                        device="0")  # Train for 3 epochs

    metrics = model.val()

    # Perform object detection on an image
    # results = model("Kitti/image.jpg")
    # results[0].show()

    # Export the model to ONNX format
    # path = model.export(format="onnx")  # return path to exported model

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
