import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, binary_dilation, binary_erosion
from scipy import ndimage
from skimage.segmentation import watershed
import flammkuchen as fl
import os
import torch


def optimized_hybrid_bb(image, spine_mask, dendrite_mask):
    """
    Optimized hybrid approach for spine instance detection
    
    Args:
        image: Original image
        spine_mask: Binary mask of spines
        dendrite_mask: Binary mask of dendrite
        
    Returns:
        tuple: (instance_labels, bounding_boxes)
    """
    # Ensure binary masks
    binary_spine = spine_mask > 0.5
    binary_dendrite = dendrite_mask > 0.5
    
    # Quick check for empty masks
    if not np.any(binary_spine):
        return np.zeros_like(binary_spine, dtype=int), []
    
    # 1. Start with dendrite-guided approach for biological context
    # Get dendrite boundary (quick skeletonization)
    skeleton = skeletonize(binary_dendrite)
    dendrite_boundary = binary_dilation(skeleton, np.ones((3, 3)))
    
    # Initial connected components
    spine_components = label(binary_spine, connectivity=2)
    
    # Initialize refined instances
    refined_instances = np.zeros_like(spine_components)
    next_id = 1
    
    # Track bounding boxes as we go
    bboxes = []
    
    # Process each connected component
    for i in range(1, spine_components.max() + 1):
        component = (spine_components == i)
        
        # Skip tiny components
        if np.sum(component) < 5:
            continue
        
        # Check if this component connects to dendrite
        connects_to_dendrite = np.any(component & dendrite_boundary)
        
        # Use different strategies based on dendrite connection
        if connects_to_dendrite:
            # Find connection points to dendrite
            connection = component & dendrite_boundary
            conn_components = label(connection)
            num_connections = conn_components.max()
            
            # Multiple connection points suggest merged spines
            if num_connections > 1 and np.sum(component) > 20:
                # Create markers at connection points
                markers = np.zeros_like(component, dtype=int)
                for j in range(1, num_connections + 1):
                    conn = (conn_components == j)
                    y, x = ndimage.center_of_mass(conn)
                    markers[int(y), int(x)] = j
                
                # Use watershed to separate based on connection points
                distance = ndimage.distance_transform_edt(component)
                separated = watershed(-distance, markers, mask=component)
                
                # Add each separated component
                for j in range(1, separated.max() + 1):
                    sub_component = (separated == j)
                    if np.sum(sub_component) > 5:
                        refined_instances[sub_component] = next_id
                        # Get bbox
                        props = regionprops(sub_component.astype(int))
                        if props:  # Ensure there are properties
                            bboxes.append(props[0].bbox)
                            next_id += 1
            else:
                # Single connection point - likely a single spine
                refined_instances[component] = next_id
                props = regionprops(component.astype(int))
                if props:
                    bboxes.append(props[0].bbox)
                    next_id += 1
        elif np.sum(component) > 10:
            # No dendrite connection, but significant size - could be spine
            # Check shape characteristics
            props = regionprops(component.astype(int))
            if props and props[0].eccentricity < 0.95:  # Not too elongated
                refined_instances[component] = next_id
                bboxes.append(props[0].bbox)
                next_id += 1
    
    # Check if we got good instance separation
    if next_id < 2 or np.sum(refined_instances > 0) < 0.5 * np.sum(binary_spine):
        # Fall back to distance-based watershed for better coverage
        distance = ndimage.distance_transform_edt(binary_spine)
        
        # Find local maxima as spine centers
        from scipy.ndimage import maximum_filter
        max_filtered = maximum_filter(distance, size=5)
        local_maxima = (distance == max_filtered) & (distance > 0.7)
        
        # If no maxima found, create a single marker
        if not np.any(local_maxima):
            y, x = ndimage.center_of_mass(binary_spine)
            local_maxima = np.zeros_like(binary_spine)
            local_maxima[int(y), int(x)] = True
        
        # Create markers for watershed
        markers, _ = ndimage.label(local_maxima)
        
        # Apply watershed
        instances = watershed(-distance, markers, mask=binary_spine)
        
        # Get bounding boxes
        props = regionprops(instances)
        bboxes = [prop.bbox for prop in props]
        
        return instances, bboxes
    
    return refined_instances, bboxes