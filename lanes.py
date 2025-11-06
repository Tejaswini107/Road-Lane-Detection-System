import cv2
import numpy as np

def make_coordinates(image, line_parameters):
    if line_parameters is None or len(line_parameters) != 2:
        return None
    slope, intercept = line_parameters
    if not np.isfinite(slope) or abs(slope) < 1e-3:
        return None
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines, min_abs_slope=0.3):
    if lines is None:
        return None

    left_fit, right_fit = [], []

    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if abs(slope) < min_abs_slope:
            continue

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    result = []

    if len(left_fit) > 0:
        left_fit_avg = np.mean(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_avg)
        if left_line is not None:
            result.append(left_line)

    if len(right_fit) > 0:
        right_fit_avg = np.mean(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_avg)
        if right_line is not None:
            result.append(right_line)

    if len(result) == 0:
        return None

    return np.array(result)

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is None:
        return line_image
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

# --- DYNAMIC REGION OF INTEREST ---
def region_of_interest(image):
    height, width = image.shape[:2]
    bottom_left = (int(width * 0.1), height)
    bottom_right = (int(width * 0.9), height)
    top_left = (int(width * 0.4), int(height * 0.6))
    top_right = (int(width * 0.6), int(height * 0.6))

    polygons = np.array([[bottom_left, bottom_right, top_right, top_left]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


cap = cv2.VideoCapture("test2.mp4")

prev_lines = None
alpha = 0.9

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    edges = canny(frame)
    cropped = region_of_interest(edges)  # dynamically scaled ROI

    lines = cv2.HoughLinesP(
        cropped,
        rho=2,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=40,
        maxLineGap=5
    )

    averaged_lines = average_slope_intercept(frame, lines)

    if averaged_lines is None:
        averaged_lines = prev_lines
    else:
        if prev_lines is not None and len(prev_lines) == len(averaged_lines):
            averaged_lines = (alpha * prev_lines + (1 - alpha) * averaged_lines).astype(int)

    line_image = display_lines(frame, averaged_lines)
    combo = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result', combo)

    prev_lines = averaged_lines

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()