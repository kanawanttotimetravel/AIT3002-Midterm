import cv2
import numpy as np
import matplotlib.pyplot as plt


from utils import get_bounding_box_from_point, extract_region


def multi_scale_template_matching(image, points, target, all_objects=[], scale_range=(0.2, 1.5), scale_steps=20, threshold=0.8):
    """
    Perform multi-scale template matching
    
    Args:
        image: Source image to search in
        template: Template image to search for
        scale_range: Tuple of (min_scale, max_scale)
        scale_steps: Number of scales to try
        threshold: Matching threshold (0-1)
    
    Returns:
        list of (x, y, scale, confidence) for matches above threshold
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
    Apply non-maximum suppression to remove overlapping matches
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