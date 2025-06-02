import cv2
import numpy as np
import pytesseract

def detect_stop_sign(img):
    # Resize the image to keep it consistent.
    img = cv2.resize(img, (640, 480))

    # Convert to HSV color space.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red color thresholds.
    lower_red1 = np.array([0, 160, 30])
    upper_red1 = np.array([5, 255, 255])
    lower_red2 = np.array([175, 160, 30])
    upper_red2 = np.array([180, 255, 255])

    # Create masks from red threshold. Need two ranges because hue wraps around.
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Apply dilation then erosion to fill in gaps. 
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in image.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if each contour could be the stop sign.
    for cnt in contours:
        # Get the area of the contour. Ignore it if the area is small.
        area = cv2.contourArea(cnt)
        if area > 100:
            # Generate an approximate polygon of the contour. Assume closed loop. Allow 4% deviation from original point positions.
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)

            # Get region of bounding rectangle. 
            x, y, w, h = cv2.boundingRect(approx)
                
            # Crop the region for OCR and converty to grayscale. OCR is text identification.
            roi = img[y:y+h, x:x+w]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Apply binary and OTSU thesholding. OTSU automatically picks best threshold based off of image histogram.
            _, roi_thresh = cv2.threshold(roi_gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Extract text from image using OCR. 
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(roi_thresh, config=custom_config)
            text = text.strip().upper()
            
            # If we found a red region AND that region is (octogonal in shape OR contains the text "STOP") then we have found a stop sign.
            if len(approx) == 8 or "STOP" in text:

                # Draw red lines between stop sign vertices.
                n = len(approx)
                for i in range(n):
                    pt1 = tuple(approx[i][0])
                    pt2 = tuple(approx[(i+1) % n][0])  # wraps around to close the shape
                    cv2.line(img, pt1, pt2, (0, 0, 255), 2)

                # Highlight stop sign vertices with magenta dots.
                cv2.drawContours(img, approx, -1, (255, 0, 255), 5)

                # Put white text on the stop sign. 
                cv2.putText(img, "STOP", (int(x+w/4), int(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 2)
                return True, mask, img

    return False, mask, img

def main():
    image_path = "/home/aidan/ENPM673_turtlebot_perception_challenge/enpm673_final_proj/enpm673_module/stop_sign.png"
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to load.")
    stop, mask, img = detect_stop_sign(img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Detected", img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()

