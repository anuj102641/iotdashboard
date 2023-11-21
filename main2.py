

import cv2
import numpy as np

def detect_objects(frame, net, classes, confidence_threshold=0.5, nms_threshold=0.3):
    # Prepare the input blob for the network
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get the network output layer names
    output_layer_names = get_output_layers(net)

    # Run forward pass to get object detections
    outs = net.forward(output_layer_names)

    # Process detections and filter by confidence and class
    detections = []
    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold and classes[class_id] == 'person':
                center_x, center_y = int(obj[0] * frame.shape[1]), int(obj[1] * frame.shape[0])
                w, h = int(obj[2] * frame.shape[1] / 2), int(obj[3] * frame.shape[0] / 2)
                x, y = center_x - w, center_y - h
                detections.append([x, y, w * 2, h * 2, confidence, class_id])


    # Apply Non-Maximum Suppression to get rid of overlapping detections
    # Apply Non-Maximum Suppression to get rid of overlapping detections
    indices = cv2.dnn.NMSBoxes([det[:4] for det in detections], [det[4] for det in detections],
                               confidence_threshold, nms_threshold)

    if len(indices) > 0 and isinstance(indices[0], np.ndarray):
        indices = indices.flatten()

    filtered_detections = [detections[i] for i in indices]

    return filtered_detections

def main():
    # Load YOLOv3 model for object detection
    model_weights = 'yolov3.weights'
    model_config = 'yolov3.cfg'
    model_classes = 'coco.names'

    net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
    with open(model_classes, 'r') as f:
        classes = f.read().strip().split('\n')

    # Replace the video file path with your stored video
    video_file_path = "input_video.mp4"
    cap = cv2.VideoCapture(r'C:\Users\102641\people count\2.mp4')

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Unable to open the video file")
        return

    # Variables for counting people
    total_people = 0
    frame_count = 0
    count_interval = 2  # Count every 2 seconds

    while True:
        # Read the next frame
        ret, frame = cap.read()

        # If frame is not read correctly or end of the video, exit
        if not ret:
            print("End of the video. Exiting...")
            break

        frame_count += 1

        # Perform YOLO object detection on the frame
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        # Detect people and get filtered detections
        detections = detect_objects(frame, net, classes)

        # Update the total count every 2 seconds
        #if frame_count == cap.get(cv2.CAP_PROP_FPS) * count_interval:
        #    total_people += len(detections)
        #    frame_count = 0

        # Draw bounding boxes around detected people
        for x, y, w, h, confidence, class_id in detections:
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{classes[int(class_id)]}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the current and total count
        cv2.putText(frame, f'Current Count: {len(detections)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #cv2.putText(frame, f'Total People: {total_people}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('People Counting', frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the video capture object
    cap.release()

    # Destroy any created windows
    cv2.destroyAllWindows()

def get_output_layers(net):
    try:
        # OpenCV 4.x
        return net.getUnconnectedOutLayersNames()
    except AttributeError:
        # OpenCV 3.x
        layer_names = net.getLayerNames()
        return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

if __name__ == "__main__":
    main()
