import cv2
import numpy as np
import matplotlib.pyplot as plt


from utils import get_bounding_box_from_point, extract_region


def multi_scale_template_matching(image, points, target, all_objects=[], scale_range=(0.2, 1.5), scale_steps=20, threshold=0.8):
    """
    Perform multi-scale template matching to find similar objects in an image.
    
    This function extracts a template from the source image using the provided points,
    then searches for similar patterns within the target region at different scales.
    It applies template matching using normalized cross-correlation and handles
    bright regions by replacing them with mean intensity.
    
    Args:
        image (numpy.ndarray): Source image to search in
        points (tuple): (x, y) coordinates where the template should be extracted
        target (tuple): Bounding box (x, y, w, h) of the region to search in
        all_objects (list, optional): List of pre-computed object bounding boxes.
            Defaults to empty list.
        scale_range (tuple, optional): (min_scale, max_scale) for template resizing.
            Defaults to (0.2, 1.5).
        scale_steps (int, optional): Number of different scales to try.
            Defaults to 20.
        threshold (float, optional): Matching confidence threshold (0-1).
            Defaults to 0.8.
    
    Returns:
        list: List of dictionaries containing match information:
            {
                'x': int,          # x coordinate of match
                'y': int,          # y coordinate of match
                'scale': float,    # scale factor of match
                'confidence': float,# matching confidence score
                'width': int,      # width of matched region
                'height': int      # height of matched region
            }
    """
    # Convert images to grayscale
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    # gray_image = cv2.GaussianBlur(gray_image,(3,3),cv2.BORDER_DEFAULT)
    
    template_bbox = get_bounding_box_from_point(gray_image, points, all_objects)
    gray_template = extract_region(gray_image, template_bbox)
    # gray_template = cv2.equalizeHist(gray_template)

    gray_image = extract_region(gray_image, target)
    # gray_image = cv2.equalizeHist(gray_image)

    mean = np.uint8(np.mean(gray_image))
    gray_template[gray_template > 250] = mean
    # gray_template = gray_template - mean
    # gray_image = gray_image - mean
    
    # Store best matches
    matches = []
    
    # Generate scales to try
    scales = np.linspace(scale_range[0], scale_range[1], scale_steps)
    
    # Try each scale
    for scale in scales:
        # Resize the image according to the scale
        resized_template = cv2.resize(gray_template, None,
                                    fx=scale, fy=scale,
                                    interpolation=cv2.INTER_AREA)
        
        # Get the current template size
        h, w = resized_template.shape
        
        # If resized template is larger than the image, skip this scale
        if h > gray_image.shape[0] or w > gray_image.shape[1]:
            continue
        
        # Apply template matching
        result = cv2.matchTemplate(gray_image, resized_template, cv2.TM_CCOEFF_NORMED)
        # plt.imshow(result, cmap='gray')
        # plt.savefig(f'plots/{scale}.png')

        # Find locations where matching exceeds the threshold
        locations = np.where(result >= threshold)
        
        for pt in zip(*locations[::-1]):  # Switch columns and rows
            match = {
                'x': int(pt[0]),
                'y': int(pt[1]),
                'scale': scale,
                'confidence': result[pt[1], pt[0]],
                'width': int(w),
                'height': int(h)
            }
            matches.append(match)
    
    # Apply non-maximum suppression to remove overlapping matches
    matches = non_max_suppression(matches)
    
    return matches


def non_max_suppression(matches, overlap_thresh=0.3):
    """
    Apply non-maximum suppression to remove overlapping matches.
    
    This function eliminates redundant overlapping detections by keeping only
    the highest confidence match in areas where multiple detections overlap
    more than the specified threshold.
    
    Args:
        matches (list): List of dictionaries containing match information:
            {
                'x': int,          # x coordinate of match
                'y': int,          # y coordinate of match
                'width': int,      # width of matched region
                'height': int,     # height of matched region
                'confidence': float # matching confidence score
            }
        overlap_thresh (float, optional): Maximum allowed overlap ratio between
            boxes (0-1). Defaults to 0.3.
    
    Returns:
        list: Filtered list of matches with overlapping detections removed,
            maintaining the same dictionary structure as input
    """
    if not matches:
        return []
    
    # Convert matches to numpy array for easier processing
    boxes = np.array([[m['x'], m['y'], m['x'] + m['width'], m['y'] + m['height']] for m in matches])
    scores = np.array([m['confidence'] for m in matches])
    
    # Calculate areas
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort by confidence
    idxs = np.argsort(scores)[::-1]
    
    pick = []
    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)
        
        # Find overlapping boxes
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        
        # Calculate overlap areas
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[idxs[1:]]
        
        # Delete overlapping boxes
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))
    
    return [matches[i] for i in pick]