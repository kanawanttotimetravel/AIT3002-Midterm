import cv2
import numpy as np
import matplotlib.pyplot as plt

def opening(image, iterations=4):
    """
    Perform morphological opening operation on an image.
    
    This function converts the image to grayscale, applies binary thresholding,
    and then performs erosion followed by dilation to remove noise and
    separate objects.
    
    Args:
        image (numpy.ndarray): Input BGR image
        iterations (int, optional): Number of iterations for erosion and dilation. 
            Defaults to 4.
    
    Returns:
        numpy.ndarray: Binary image after morphological opening
    """
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _ ,thresh = cv2.threshold(gray ,250,255,cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    erosion = cv2.erode(thresh, kernel, iterations=iterations)
    dilation = cv2.dilate(erosion, kernel, iterations=iterations)

    return dilation


def get_all_object(image, all_objects=[]):
    """
    Detect and return bounding boxes for all large objects in the image.
    
    This function applies morphological opening and contour detection to find
    objects in the image. Only objects with width or height >= 100 pixels
    are included.
    
    Args:
        image (numpy.ndarray): Input BGR image
        all_objects (list, optional): Cache of previously detected objects.
            If provided and not empty, returns this instead of recomputing.
            Defaults to empty list.
    
    Returns:
        list: List of tuples (x, y, width, height) representing bounding boxes
            of detected objects
    """
    if len(all_objects):
        return all_objects
    opened = opening(image)
    contours, _ = cv2.findContours(opened,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    img = opened.copy()
    bounding_boxes = []
    for contour in contours:    
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
        if max(w, h) >= 100:
            bounding_boxes.append((x,y,w,h))

    print('Got all objects')
    return bounding_boxes


def get_bounding_box_from_point(image, points, all_objects=[]):
    """
    Find the bounding box of an object containing the given point.
    
    Args:
        image (numpy.ndarray): Input BGR image
        points (tuple): Point coordinates (x, y) to check
        all_objects (list, optional): List of pre-computed object bounding boxes.
            If empty, will compute objects using get_all_object().
            Defaults to empty list.
    
    Returns:
        tuple or None: Bounding box as (x, y, width, height) if point is inside
            an object, None otherwise
    """
    for object in get_all_object(image, all_objects):
        x, y, w, h = object
        u, v = points
        if u >= x and v >= y and u <= x + w and v <= y + h:
            return object
    return None
    

def extract_region(image, bbox):
    """
    Extract a region from an image using a bounding box
    
    Args:
        image: Input image (numpy array)
        bbox: Tuple of (x, y, width, height) or (x1, y1, x2, y2)
    
    Returns:
        cropped_image: The extracted region
    """
    x, y, w, h = bbox
    cropped = image[y:y+h, x:x+w]
    
    return cropped