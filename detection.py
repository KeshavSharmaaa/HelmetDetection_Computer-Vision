import cv2
import numpy as np
import sys
import os


def detect_helmet(frame):
    output = frame.copy()
    h_img, w_img = frame.shape[:2]

    # ── 1. Focus on the upper 60% of the image (where a helmet would be) ──
    upper = frame[:int(h_img * 0.65), :]

    # ── 2. Upscale if image is small (improves contour quality) ──
    scale = max(1, 600 // max(h_img, w_img))
    if scale > 1:
        upper_big = cv2.resize(upper, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LINEAR)
    else:
        upper_big = upper.copy()

    gray = cv2.cvtColor(upper_big, cv2.COLOR_BGR2GRAY)

    # ── 3. Edge detection on upscaled crop ──
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 30, 100)

    kernel = np.ones((7, 7), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # ── 4. Relaxed HSV check for dark/black helmet colors ──
    def is_helmet_color(roi):
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Black / very dark
        mask_black = cv2.inRange(hsv, (0, 0, 0), (180, 255, 90))
        # White / light gray (visor)
        mask_white = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
        # Any strong saturated color (red, blue, yellow helmets)
        mask_color = cv2.inRange(hsv, (0, 80, 80), (180, 255, 255))
        combined = cv2.bitwise_or(mask_black, cv2.bitwise_or(mask_white, mask_color))
        return (np.sum(combined > 0) / combined.size) > 0.25   # lowered to 25%

    helmet_detected = False
    best_box = None   # track the most circular candidate

    UH, UW = upper_big.shape[:2]

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Area bounds scaled with image size
        min_area = (UH * UW) * 0.02    # at least 2 % of the crop
        max_area = (UH * UW) * 0.70    # no more than 70 %
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # ── Relaxed: 0.20 covers oval/angled helmets ──
        if circularity < 0.20:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Aspect ratio: helmet should be roughly square-ish
        aspect = w / h if h > 0 else 0
        if aspect < 0.5 or aspect > 2.2:
            continue

        roi = upper_big[y:y+h, x:x+w]
        if not is_helmet_color(roi):
            continue

        # Map coordinates back to original frame
        sx = x // scale
        sy = y // scale
        sw = w // scale
        sh = h // scale

        helmet_detected = True
        best_box = (sx, sy, sw, sh)
        cv2.rectangle(output, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
        cv2.putText(output, f"Helmet ({circularity:.2f})",
                    (sx, max(sy - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # ── Fallback: if nothing passed the full filter, try a simpler dark-blob pass ──
    if not helmet_detected:
        helmet_detected, output = _fallback_dark_blob(frame, output)

    label = "Helmet Detected" if helmet_detected else "No Helmet"
    color = (0, 255, 0) if helmet_detected else (0, 0, 255)
    cv2.putText(output, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return output, helmet_detected


def _fallback_dark_blob(frame, output):
    """
    Simpler fallback: look for a large dark rounded blob in the upper frame.
    Works well for black helmets with subtle edges.
    """
    h_img, w_img = frame.shape[:2]
    upper = frame[:int(h_img * 0.6), :]

    hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))  # dark pixels

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    UH, UW = upper.shape[:2]
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        area = cv2.contourArea(cnt)
        if area < (UH * UW) * 0.03 or area > (UH * UW) * 0.75:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h > 0 else 0
        if aspect < 0.4 or aspect > 2.5:
            continue

        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 200, 255), 2)
        cv2.putText(output, "Helmet (fallback)", (x, max(y - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
        return True, output

    return False, output


def process_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Could not read image: {image_path}")
        return

    print(f"✅ Loaded: {frame.shape[1]}×{frame.shape[0]} px")

    result, detected = detect_helmet(frame)

    print("🪖 Helmet DETECTED!" if detected else "❌ No helmet found.")

    output_path = "output_" + os.path.basename(image_path)
    cv2.imwrite(output_path, result)
    print(f"💾 Saved: {output_path}")

    # Uncomment if running locally with a display:
    # cv2.imshow("Helmet Detection", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return output_path


if __name__ == "__main__":
    DEFAULT_IMAGE = "train_img1.jpg"
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    process_image(path)