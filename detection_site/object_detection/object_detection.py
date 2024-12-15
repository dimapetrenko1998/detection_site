import cv2
import numpy as np


def load_mobilenet_model():
    # Загрузка модели MobileNet SSD
    model = cv2.dnn.readNetFromCaffe('mobilenet_ssd.prototxt', 'mobilenet_ssd.caffemodel')
    return model


def load_yolo_model():
    # Загрузка модели YOLOv3
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    with open('coco.names', 'r') as f:
        classes = f.read().strip().split('\n')
    return net, classes


def detect_objects_with_mobilenet(image_path):
    """
    Обнаруживает объекты на изображении с использованием MobileNet SSD.

    Args:
        image_path (str): Путь к изображению.

    Returns:
        List[Dict[str, Union[str, float]]]: Список обнаруженных объектов с их классами и уверенностью.
    """
    model = load_mobilenet_model()
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Создание блоба из изображения
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    model.setInput(blob)
    detections = model.forward()

    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # Порог уверенности
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            results.append({"class": idx, "confidence": confidence, "box": (startX, startY, endX, endY)})
    return results


def detect_objects_with_yolo(image_path):
    """
    Обнаруживает объекты на изображении с использованием YOLOv3.

    Args:
        image_path (str): Путь к изображению.

    Returns:
        List[Dict[str, Union[str, float]]]: Список обнаруженных объектов с их классами и уверенностью.
    """
    net, classes = load_yolo_model()
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Создание блоба из изображения
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    results = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                results.append({"class": classes[class_id], "confidence": confidence, "box": (x, y, x + w, y + h)})
    return results
