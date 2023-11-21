import cv2

def main():
    # Initialize video capture objects for three cameras.
    cap1 = cv2.VideoCapture(r'C:\Users\102641\people count\1st angle.mp4')
    cap2 = cv2.VideoCapture(r'C:\Users\102641\people count\2.mp4')
    cap3 = cv2.VideoCapture(r'C:\Users\102641\people count\output.mp4')

    # Check if cameras are opened successfully.
    if not cap1.isOpened() or not cap2.isOpened() or not cap3.isOpened():
        print("Error: One or more cameras couldn't be opened.")
        return

    # Set common resolution (adjust as needed).
    width, height = 640, 480
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap3.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while True:
        # Read frames from each camera.
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()

        # Check if all frames were read successfully.
        if not ret1 or not ret2 or not ret3:
            print("Error: Failed to capture a frame.")
            break

        # Combine the frames horizontally (side-by-side).
        combined_frame = cv2.hconcat([frame1, frame2, frame3])

        # Display the combined frame.
        cv2.imshow('Stitched Video', combined_frame)

        # Exit the loop when 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture objects and close the display window.
    cap1.release()
    cap2.release()
    cap3.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
