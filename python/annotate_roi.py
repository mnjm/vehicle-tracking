import cv2
import sys
from pathlib import Path

def draw_bbox(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    cap.release()

    # Let user select ROI
    bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if bbox == (0, 0, 0, 0):
        print("No ROI selected.")
        return

    # Draw the selected bounding box
    x, y, w, h = map(int, bbox)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save bbox coordinates
    video_path = Path(video_path)
    roi_file = str(video_path.parent / video_path.stem) + "_roi.txt"
    with open(roi_file, "w") as f:
        f.write(f"{x} {y} {x+w} {y+h}\n")

    print(f"ROI saved to {roi_file}")

    # Display frame with bbox
    cv2.imshow("Annotated Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = sys.argv[1]
    draw_bbox(video_path)
