import numpy as np
import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import tifffile
from tqdm import tqdm
import matplotlib.cm as cm
from torchvision.ops import nms
import json
from datetime import datetime
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.feature import graycomatrix, graycoprops
from scipy.spatial.distance import cdist
from collections import defaultdict

def load_tif_stack(tif_path):
    img = Image.open(tif_path)
    slices = []
    try:
        while True:
            slices.append(np.array(img.copy()))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return np.stack(slices)

def normalize_patch(patch):
    patch = patch.astype(np.float32)
    if patch.max() > patch.min():
        patch = (patch - patch.min()) / (patch.max() - patch.min())
    patch_uint8 = (patch * 255).astype(np.uint8)
    return cv2.cvtColor(patch_uint8, cv2.COLOR_GRAY2RGB)

def extract_patch_safe(slice_array, x, y, patch_size):
    """Extract patch with padding for edge cases"""
    h, w = slice_array.shape
    patch = np.zeros((patch_size, patch_size), dtype=slice_array.dtype)
    
    x_end = min(x + patch_size, w)
    y_end = min(y + patch_size, h)
    
    patch[:y_end-y, :x_end-x] = slice_array[y:y_end, x:x_end]
    return patch

def get_patch_positions(h, w, patch_size, stride):
    y_steps = list(range(0, h - patch_size + 1, stride))
    x_steps = list(range(0, w - patch_size + 1, stride))
    if y_steps[-1] + patch_size < h:
        y_steps.append(h - patch_size)
    if x_steps[-1] + patch_size < w:
        x_steps.append(w - patch_size)
    return [(x, y) for y in y_steps for x in x_steps]

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj

# Size filtering class (from previous implementation)
class SizeFilter:
    """Size-based filtering for YOLO detections based on ground truth analysis"""
    
    def __init__(self, filtering_mode='conservative'):
        self.thresholds = {
            'conservative': {
                'min_bbox_area': 28,
                'max_bbox_area': 396,
                'min_width': 5,
                'max_width': 21,
                'min_height': 5,
                'max_height': 21,
                'max_aspect_ratio': 1.8,
                'min_aspect_ratio': 1.0 / 1.8
            },
            'aggressive': {
                'min_bbox_area': 42,
                'max_bbox_area': 289,
                'min_width': 5,
                'max_width': 21,
                'min_height': 5,
                'max_height': 21,
                'max_aspect_ratio': 1.4,
                'min_aspect_ratio': 1.0 / 1.4
            }
        }
        
        if isinstance(filtering_mode, dict):
            self.current_thresholds = filtering_mode
        else:
            self.current_thresholds = self.thresholds[filtering_mode]
        
        self.filtering_mode = filtering_mode
        self.reset_stats()
    
    def filter_detections(self, boxes, scores, debug=False):
        if len(boxes) == 0:
            return boxes, scores, np.array([], dtype=bool)
        
        # Calculate box properties
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        aspect_ratios = np.maximum(widths / np.maximum(heights, 1e-6), 
                                  heights / np.maximum(widths, 1e-6))
        
        # Apply filters
        filter_mask = np.ones(len(boxes), dtype=bool)
        
        # Area filtering
        area_mask = (areas >= self.current_thresholds['min_bbox_area']) & \
                   (areas <= self.current_thresholds['max_bbox_area'])
        filter_mask &= area_mask
        
        # Dimension filtering
        width_mask = (widths >= self.current_thresholds['min_width']) & \
                    (widths <= self.current_thresholds['max_width'])
        height_mask = (heights >= self.current_thresholds['min_height']) & \
                     (heights <= self.current_thresholds['max_height'])
        dimension_mask = width_mask & height_mask
        filter_mask &= dimension_mask
        
        # Aspect ratio filtering
        aspect_mask = (aspect_ratios <= self.current_thresholds['max_aspect_ratio']) & \
                     (aspect_ratios >= self.current_thresholds['min_aspect_ratio'])
        filter_mask &= aspect_mask
        
        return boxes[filter_mask], scores[filter_mask], filter_mask
    
    def reset_stats(self):
        self.stats = {'total': 0, 'passed': 0}

# Intensity Filter Class (from previous implementation)
class IntensityFilter:
    """Intensity-based filtering for YOLO detections analyzing raw pixel intensities"""
    
    def __init__(self, filtering_mode='moderate'):
        self.thresholds = {
            'strict': {
                'min_mean_intensity_ratio': 1.15,
                'max_mean_intensity_ratio': 4.0,
                'min_contrast_ratio': 0.2,
                'max_intensity_std_ratio': 0.8,
                'min_edge_strength': 0.1,
                'min_signal_to_noise': 2.0,
                'background_percentile': 25
            },
            'moderate': {
                'min_mean_intensity_ratio': 1.1,
                'max_mean_intensity_ratio': 5.0,
                'min_contrast_ratio': 0.15,
                'max_intensity_std_ratio': 1.0,
                'min_edge_strength': 0.08,
                'min_signal_to_noise': 1.5,
                'background_percentile': 30
            },
            'loose': {
                'min_mean_intensity_ratio': 1.05,
                'max_mean_intensity_ratio': 6.0,
                'min_contrast_ratio': 0.1,
                'max_intensity_std_ratio': 1.2,
                'min_edge_strength': 0.05,
                'min_signal_to_noise': 1.2,
                'background_percentile': 35
            }
        }
        
        if isinstance(filtering_mode, dict):
            self.current_thresholds = filtering_mode
        else:
            self.current_thresholds = self.thresholds[filtering_mode]
        
        self.filtering_mode = filtering_mode
        self.reset_stats()
    
    def calculate_local_background(self, image, box, margin=10):
        x1, y1, x2, y2 = map(int, box)
        h, w = image.shape
        
        bg_x1 = max(0, x1 - margin)
        bg_y1 = max(0, y1 - margin)
        bg_x2 = min(w, x2 + margin)
        bg_y2 = min(h, y2 + margin)
        
        bg_region = image[bg_y1:bg_y2, bg_x1:bg_x2].copy()
        
        obj_x1_rel = max(0, x1 - bg_x1)
        obj_y1_rel = max(0, y1 - bg_y1)
        obj_x2_rel = min(bg_x2 - bg_x1, x2 - bg_x1)
        obj_y2_rel = min(bg_y2 - bg_y1, y2 - bg_y1)
        
        bg_mask = np.ones_like(bg_region, dtype=bool)
        if obj_x2_rel > obj_x1_rel and obj_y2_rel > obj_y1_rel:
            bg_mask[obj_y1_rel:obj_y2_rel, obj_x1_rel:obj_x2_rel] = False
        
        if np.any(bg_mask):
            bg_values = bg_region[bg_mask]
            background_intensity = np.percentile(bg_values, self.current_thresholds['background_percentile'])
        else:
            background_intensity = np.percentile(image, self.current_thresholds['background_percentile'])
        
        return background_intensity
    
    def calculate_edge_strength(self, image, box):
        x1, y1, x2, y2 = map(int, box)
        margin = 2
        h, w = image.shape
        ext_x1 = max(0, x1 - margin)
        ext_y1 = max(0, y1 - margin)
        ext_x2 = min(w, x2 + margin)
        ext_y2 = min(h, y2 + margin)
        
        region = image[ext_y1:ext_y2, ext_x1:ext_x2]
        
        grad_x = np.abs(cv2.Sobel(region.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
        grad_y = np.abs(cv2.Sobel(region.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3))
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        if region.max() > region.min():
            edge_strength = np.mean(gradient_magnitude) / (region.max() - region.min())
        else:
            edge_strength = 0.0
        
        return edge_strength
    
    def analyze_object_intensity(self, image, box):
        x1, y1, x2, y2 = map(int, box)
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        mean_intensity = np.mean(roi)
        std_intensity = np.std(roi)
        min_intensity = np.min(roi)
        max_intensity = np.max(roi)
        
        background_intensity = self.calculate_local_background(image, box)
        intensity_ratio = mean_intensity / max(background_intensity, 1e-6)
        
        if mean_intensity > 0:
            contrast_ratio = (max_intensity - min_intensity) / mean_intensity
            std_ratio = std_intensity / mean_intensity
        else:
            contrast_ratio = 0
            std_ratio = 0
        
        noise_estimate = max(std_intensity, np.std(image) * 0.1)
        signal_to_noise = (mean_intensity - background_intensity) / max(noise_estimate, 1e-6)
        
        edge_strength = self.calculate_edge_strength(image, box)
        
        features = {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'min_intensity': min_intensity,
            'max_intensity': max_intensity,
            'background_intensity': background_intensity,
            'intensity_ratio': intensity_ratio,
            'contrast_ratio': contrast_ratio,
            'std_ratio': std_ratio,
            'signal_to_noise': signal_to_noise,
            'edge_strength': edge_strength
        }
        
        return features
    
    def filter_detections(self, boxes, scores, image, debug=False):
        if len(boxes) == 0:
            return boxes, scores, np.array([], dtype=bool), []
        
        filter_mask = np.ones(len(boxes), dtype=bool)
        intensity_features_list = []
        
        for i, box in enumerate(boxes):
            features = self.analyze_object_intensity(image, box)
            intensity_features_list.append(features)
            
            if features is None:
                filter_mask[i] = False
                continue
            
            passed_filters = True
            
            if not (self.current_thresholds['min_mean_intensity_ratio'] <= 
                   features['intensity_ratio'] <= 
                   self.current_thresholds['max_mean_intensity_ratio']):
                passed_filters = False
            
            if features['contrast_ratio'] < self.current_thresholds['min_contrast_ratio']:
                passed_filters = False
            
            if features['std_ratio'] > self.current_thresholds['max_intensity_std_ratio']:
                passed_filters = False
            
            if features['edge_strength'] < self.current_thresholds['min_edge_strength']:
                passed_filters = False
            
            if features['signal_to_noise'] < self.current_thresholds['min_signal_to_noise']:
                passed_filters = False
            
            filter_mask[i] = passed_filters
        
        return (boxes[filter_mask], scores[filter_mask], 
                filter_mask, [intensity_features_list[i] for i in range(len(boxes)) if filter_mask[i]])
    
    def reset_stats(self):
        self.stats = {'total': 0, 'passed': 0}

# New Plane Coverage Filter Class
class PlaneCoverageFilter:
    """3D plane coverage filtering for analyzing detection continuity across z-slices"""
    
    def __init__(self, filtering_mode='moderate'):
        """
        Initialize plane coverage filter
        
        Args:
            filtering_mode: 'strict', 'moderate', 'loose', or custom dict with thresholds
        """
        
        self.thresholds = {
            'strict': {
                'min_plane_coverage': 3,           # Must appear in at least 3 slices
                'max_spatial_distance': 15,        # Max distance between detections in adjacent slices
                'min_continuity_ratio': 0.6,       # Must be present in 60% of expected slices
                'max_gap_size': 1,                  # Max gap in slice sequence
                'edge_slice_tolerance': 2,          # Allow fewer requirements near volume edges
                'min_track_length': 3,              # Minimum track length for validation
                'spatial_consistency_weight': 0.3   # Weight for spatial consistency scoring
            },
            'moderate': {
                'min_plane_coverage': 2,           # Must appear in at least 2 slices
                'max_spatial_distance': 20,        # More lenient spatial matching
                'min_continuity_ratio': 0.4,       # 40% continuity required
                'max_gap_size': 2,                  # Allow 2-slice gaps
                'edge_slice_tolerance': 3,
                'min_track_length': 2,
                'spatial_consistency_weight': 0.2
            },
            'loose': {
                'min_plane_coverage': 1,           # Single slice detections allowed
                'max_spatial_distance': 25,
                'min_continuity_ratio': 0.2,       # 20% continuity
                'max_gap_size': 3,                  # Allow 3-slice gaps
                'edge_slice_tolerance': 4,
                'min_track_length': 1,
                'spatial_consistency_weight': 0.1
            }
        }
        
        if isinstance(filtering_mode, dict):
            self.current_thresholds = filtering_mode
        else:
            self.current_thresholds = self.thresholds[filtering_mode]
        
        self.filtering_mode = filtering_mode
        self.reset_stats()
        
        # Storage for cross-slice analysis
        self.all_slice_detections = []  # List of detections per slice
        self.detection_tracks = []       # 3D tracks of detections
        
        print(f"Plane coverage filter initialized in {filtering_mode} mode:")
        print(f"  Min plane coverage: {self.current_thresholds['min_plane_coverage']} slices")
        print(f"  Max spatial distance: {self.current_thresholds['max_spatial_distance']} pixels")
        print(f"  Min continuity ratio: {self.current_thresholds['min_continuity_ratio']:.1f}")
    
    def store_slice_detections(self, slice_idx, boxes, scores, features=None):
        """
        Store detections for a slice for later 3D analysis
        
        Args:
            slice_idx: Index of the current slice
            boxes: numpy array of [x1, y1, x2, y2] boxes
            scores: numpy array of confidence scores
            features: optional intensity features
        """
        # Calculate centroids
        if len(boxes) > 0:
            centroids = np.column_stack([
                (boxes[:, 0] + boxes[:, 2]) / 2,  # x center
                (boxes[:, 1] + boxes[:, 3]) / 2   # y center
            ])
        else:
            centroids = np.array([]).reshape(0, 2)
        
        slice_data = {
            'slice_idx': slice_idx,
            'boxes': boxes.copy() if len(boxes) > 0 else np.array([]),
            'scores': scores.copy() if len(scores) > 0 else np.array([]),
            'centroids': centroids,
            'features': features if features else [],
            'n_detections': len(boxes)
        }
        
        # Ensure we have enough slots in the list
        while len(self.all_slice_detections) <= slice_idx:
            self.all_slice_detections.append(None)
        
        self.all_slice_detections[slice_idx] = slice_data
    
    def build_detection_tracks(self, total_slices):
        """
        Build 3D tracks by linking detections across slices
        
        Args:
            total_slices: Total number of slices in the volume
            
        Returns:
            tracks: List of detection tracks across slices
        """
        tracks = []
        
        # Process slices sequentially to build tracks
        for slice_idx in range(total_slices):
            if (slice_idx >= len(self.all_slice_detections) or 
                self.all_slice_detections[slice_idx] is None):
                continue
            
            current_slice = self.all_slice_detections[slice_idx]
            current_centroids = current_slice['centroids']
            
            if len(current_centroids) == 0:
                continue
            
            # For first slice, start new tracks
            if slice_idx == 0:
                for i, centroid in enumerate(current_centroids):
                    track = {
                        'track_id': len(tracks),
                        'slices': [slice_idx],
                        'centroids': [centroid],
                        'boxes': [current_slice['boxes'][i]],
                        'scores': [current_slice['scores'][i]],
                        'features': [current_slice['features'][i] if current_slice['features'] else None],
                        'start_slice': slice_idx,
                        'end_slice': slice_idx,
                        'length': 1
                    }
                    tracks.append(track)
            else:
                # Try to extend existing tracks
                unmatched_detections = list(range(len(current_centroids)))
                
                for track in tracks:
                    if track['end_slice'] < slice_idx - self.current_thresholds['max_gap_size'] - 1:
                        continue  # Track is too old to extend
                    
                    # Find closest detection to extend this track
                    last_centroid = track['centroids'][-1]
                    
                    if len(unmatched_detections) > 0:
                        distances = cdist([last_centroid], 
                                        current_centroids[unmatched_detections])[0]
                        
                        min_dist_idx = np.argmin(distances)
                        min_distance = distances[min_dist_idx]
                        
                        if min_distance <= self.current_thresholds['max_spatial_distance']:
                            # Extend track
                            detection_idx = unmatched_detections[min_dist_idx]
                            track['slices'].append(slice_idx)
                            track['centroids'].append(current_centroids[detection_idx])
                            track['boxes'].append(current_slice['boxes'][detection_idx])
                            track['scores'].append(current_slice['scores'][detection_idx])
                            track['features'].append(current_slice['features'][detection_idx] 
                                                   if current_slice['features'] else None)
                            track['end_slice'] = slice_idx
                            track['length'] = len(track['slices'])
                            
                            # Remove matched detection
                            unmatched_detections.remove(detection_idx)
                
                # Start new tracks for unmatched detections
                for detection_idx in unmatched_detections:
                    track = {
                        'track_id': len(tracks),
                        'slices': [slice_idx],
                        'centroids': [current_centroids[detection_idx]],
                        'boxes': [current_slice['boxes'][detection_idx]],
                        'scores': [current_slice['scores'][detection_idx]],
                        'features': [current_slice['features'][detection_idx] 
                                   if current_slice['features'] else None],
                        'start_slice': slice_idx,
                        'end_slice': slice_idx,
                        'length': 1
                    }
                    tracks.append(track)
        
        return tracks
    
    def analyze_track_quality(self, track, total_slices):
        """
        Analyze the quality of a detection track
        
        Args:
            track: Detection track dictionary
            total_slices: Total number of slices in volume
            
        Returns:
            quality_metrics: Dictionary of track quality metrics
        """
        track_length = track['length']
        slice_span = track['end_slice'] - track['start_slice'] + 1
        
        # Calculate continuity ratio
        if slice_span > 0:
            continuity_ratio = track_length / slice_span
        else:
            continuity_ratio = 1.0
        
        # Calculate spatial consistency (how much the detection moves)
        spatial_consistency = 1.0
        if len(track['centroids']) > 1:
            movements = []
            for i in range(1, len(track['centroids'])):
                movement = np.linalg.norm(
                    np.array(track['centroids'][i]) - np.array(track['centroids'][i-1])
                )
                movements.append(movement)
            
            avg_movement = np.mean(movements)
            max_movement = np.max(movements)
            
            # Penalize large movements (normalize by max allowed distance)
            spatial_consistency = 1.0 - min(avg_movement / self.current_thresholds['max_spatial_distance'], 1.0)
        
        # Calculate confidence consistency
        confidence_consistency = 1.0
        if len(track['scores']) > 1:
            score_std = np.std(track['scores'])
            score_mean = np.mean(track['scores'])
            if score_mean > 0:
                confidence_consistency = 1.0 - min(score_std / score_mean, 1.0)
        
        # Edge proximity bonus/penalty
        edge_proximity = 0.0
        near_start_edge = track['start_slice'] < self.current_thresholds['edge_slice_tolerance']
        near_end_edge = track['end_slice'] > total_slices - self.current_thresholds['edge_slice_tolerance'] - 1
        
        if near_start_edge or near_end_edge:
            edge_proximity = 0.1  # Small bonus for edge tracks
        
        # Calculate gaps in the track
        expected_slices = set(range(track['start_slice'], track['end_slice'] + 1))
        actual_slices = set(track['slices'])
        gap_slices = expected_slices - actual_slices
        gap_count = len(gap_slices)
        max_gap_size = 0
        
        if gap_slices:
            # Find maximum consecutive gap
            sorted_gaps = sorted(gap_slices)
            current_gap = 1
            max_gap_size = 1
            
            for i in range(1, len(sorted_gaps)):
                if sorted_gaps[i] == sorted_gaps[i-1] + 1:
                    current_gap += 1
                    max_gap_size = max(max_gap_size, current_gap)
                else:
                    current_gap = 1
        
        quality_metrics = {
            'track_length': track_length,
            'slice_span': slice_span,
            'continuity_ratio': continuity_ratio,
            'spatial_consistency': spatial_consistency,
            'confidence_consistency': confidence_consistency,
            'edge_proximity': edge_proximity,
            'gap_count': gap_count,
            'max_gap_size': max_gap_size,
            'avg_confidence': float(np.mean(track['scores'])),
            'near_edge': near_start_edge or near_end_edge
        }
        
        return quality_metrics
    
    def filter_tracks_by_coverage(self, tracks, total_slices):
        """
        Filter tracks based on plane coverage criteria
        
        Args:
            tracks: List of detection tracks
            total_slices: Total number of slices
            
        Returns:
            valid_tracks, filtered_tracks, track_quality_metrics
        """
        valid_tracks = []
        filtered_tracks = []
        track_quality_metrics = []
        
        for track in tracks:
            quality = self.analyze_track_quality(track, total_slices)
            track_quality_metrics.append(quality)
            
            # Apply filtering criteria
            passed_filters = True
            filter_reasons = []
            
            # 1. Minimum plane coverage
            if quality['track_length'] < self.current_thresholds['min_plane_coverage']:
                # Exception for edge tracks
                if not quality['near_edge'] or quality['track_length'] < 1:
                    passed_filters = False
                    filter_reasons.append('insufficient_plane_coverage')
            
            # 2. Continuity ratio
            if quality['continuity_ratio'] < self.current_thresholds['min_continuity_ratio']:
                passed_filters = False
                filter_reasons.append('poor_continuity')
            
            # 3. Maximum gap size
            if quality['max_gap_size'] > self.current_thresholds['max_gap_size']:
                passed_filters = False
                filter_reasons.append('excessive_gaps')
            
            # 4. Minimum track length
            if quality['track_length'] < self.current_thresholds['min_track_length']:
                passed_filters = False
                filter_reasons.append('track_too_short')
            
            # Store filtering reason
            quality['passed_filters'] = passed_filters
            quality['filter_reasons'] = filter_reasons
            
            if passed_filters:
                valid_tracks.append(track)
                self.stats['tracks_passed'] += 1
            else:
                filtered_tracks.append(track)
                for reason in filter_reasons:
                    self.stats[f'filtered_by_{reason}'] += 1
            
            self.stats['total_tracks'] += 1
        
        return valid_tracks, filtered_tracks, track_quality_metrics
    
    def apply_plane_coverage_filtering(self, total_slices):
        """
        Apply 3D plane coverage filtering to all stored detections
        
        Args:
            total_slices: Total number of slices in the volume
            
        Returns:
            filtered_detections_per_slice: List of filtered detections for each slice
            track_analysis: Comprehensive track analysis results
        """
        print(f"Applying plane coverage filtering across {total_slices} slices...")
        
        # Build 3D tracks
        tracks = self.build_detection_tracks(total_slices)
        print(f"Built {len(tracks)} detection tracks")
        
        # Filter tracks by coverage criteria
        valid_tracks, filtered_tracks, track_quality = self.filter_tracks_by_coverage(tracks, total_slices)
        print(f"Valid tracks: {len(valid_tracks)}, Filtered tracks: {len(filtered_tracks)}")
        
        # Create filtered detection lists for each slice
        filtered_detections_per_slice = [None] * total_slices
        
        for slice_idx in range(total_slices):
            filtered_detections_per_slice[slice_idx] = {
                'boxes': [],
                'scores': [],
                'features': [],
                'track_ids': [],
                'quality_metrics': []
            }
        
        # Populate filtered detections from valid tracks
        for track in valid_tracks:
            track_id = track['track_id']
            
            for i, slice_idx in enumerate(track['slices']):
                if slice_idx < total_slices:
                    filtered_detections_per_slice[slice_idx]['boxes'].append(track['boxes'][i])
                    filtered_detections_per_slice[slice_idx]['scores'].append(track['scores'][i])
                    filtered_detections_per_slice[slice_idx]['features'].append(track['features'][i])
                    filtered_detections_per_slice[slice_idx]['track_ids'].append(track_id)
                    
                    # Find corresponding quality metrics
                    track_quality_metric = None
                    for quality in track_quality:
                        if tracks[track_id] == track:  # Match by reference
                            track_quality_metric = quality
                            break
                    filtered_detections_per_slice[slice_idx]['quality_metrics'].append(track_quality_metric)
        
        # Convert lists to numpy arrays
        for slice_idx in range(total_slices):
            slice_data = filtered_detections_per_slice[slice_idx]
            if slice_data['boxes']:
                slice_data['boxes'] = np.array(slice_data['boxes'])
                slice_data['scores'] = np.array(slice_data['scores'])
            else:
                slice_data['boxes'] = np.array([]).reshape(0, 4)
                slice_data['scores'] = np.array([])
        
        # Comprehensive track analysis
        track_analysis = {
            'total_tracks': len(tracks),
            'valid_tracks': len(valid_tracks),
            'filtered_tracks': len(filtered_tracks),
            'tracks_data': tracks,
            'valid_tracks_data': valid_tracks,
            'filtered_tracks_data': filtered_tracks,
            'track_quality_metrics': track_quality,
            'filtering_summary': self.get_filtering_stats()
        }
        
        return filtered_detections_per_slice, track_analysis
    
    def get_filtering_stats(self):
        """Get comprehensive plane coverage filtering statistics"""
        if self.stats['total_tracks'] == 0:
            return self.stats
        
        stats_with_percentages = {}
        total = self.stats['total_tracks']
        
        # Convert all stats to native Python types for JSON serialization
        for key, value in self.stats.items():
            if isinstance(value, (np.integer, np.int64, np.int32)):
                stats_with_percentages[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                stats_with_percentages[key] = float(value)
            else:
                stats_with_percentages[key] = value
        
        # Calculate percentages as native Python floats
        for key in self.stats:
            if key.startswith('filtered_by_'):
                rate_key = key.replace('filtered_by_', 'filter_rate_')
                stats_with_percentages[rate_key] = float(self.stats[key] / total * 100)
        
        stats_with_percentages['pass_rate'] = float(self.stats['tracks_passed'] / total * 100)
        
        return stats_with_percentages
    
    def reset_stats(self):
        """Reset filtering statistics"""
        self.stats = {
            'total_tracks': 0,
            'tracks_passed': 0,
            'filtered_by_insufficient_plane_coverage': 0,
            'filtered_by_poor_continuity': 0,
            'filtered_by_excessive_gaps': 0,
            'filtered_by_track_too_short': 0
        }

# Color mapping functions
cmap = cm.get_cmap('viridis')

def confidence_to_color(conf):
    r, g, b, _ = cmap(conf)
    return int(r * 255), int(g * 255), int(b * 255)

def create_color_legend(width, height):
    legend = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        conf = 1.0 - i / height
        color = confidence_to_color(conf)
        legend[i, :] = color
    return legend

def run_yolo_with_comprehensive_filtering(tif_path, model_path, patch_size=192, stride=None,
                                        conf_thresh=0.25, nms_thresh=0.5, batch_size=8,
                                        size_filtering_mode='conservative',
                                        intensity_filtering_mode='moderate',
                                        plane_coverage_mode='moderate',
                                        save_path="comprehensive_filtered_output.tif",
                                        save_stats=True):
    """
    Complete YOLO inference pipeline with size, intensity, and plane coverage filtering
    
    Args:
        tif_path: Path to input TIF stack
        model_path: Path to YOLO model
        patch_size: Size of sliding window patches
        stride: Stride for sliding window (None = patch_size//2)
        conf_thresh: Confidence threshold for initial filtering
        nms_thresh: NMS IoU threshold
        batch_size: Batch size for inference
        size_filtering_mode: 'conservative', 'aggressive', or custom thresholds dict
        intensity_filtering_mode: 'strict', 'moderate', 'loose', or custom thresholds dict
        plane_coverage_mode: 'strict', 'moderate', 'loose', or custom thresholds dict
        save_path: Output path for filtered results
        save_stats: Whether to save filtering statistics
        
    Returns:
        Statistics dictionary with comprehensive filtering results
    """
    
    # Initialize components
    model = YOLO(model_path)
    size_filter = SizeFilter(size_filtering_mode)
    intensity_filter = IntensityFilter(intensity_filtering_mode)
    plane_filter = PlaneCoverageFilter(plane_coverage_mode)
    tif_stack = load_tif_stack(tif_path)
    n_slices, h, w = tif_stack.shape
    
    # Adaptive stride
    if stride is None:
        stride = patch_size // 2
    
    print(f"Loaded {n_slices} slices of size {h}x{w}")
    print(f"Patch size: {patch_size}, Stride: {stride}, Batch size: {batch_size}")
    print(f"Size filtering mode: {size_filtering_mode}")
    print(f"Intensity filtering mode: {intensity_filtering_mode}")
    print(f"Plane coverage mode: {plane_coverage_mode}")
    
    slice_statistics = []
    
    # PHASE 1: Process each slice individually (size + intensity filtering)
    print("\n=== PHASE 1: Individual slice processing ===")
    
    for slice_idx in tqdm(range(n_slices), desc="Processing individual slices"):
        slice_array = tif_stack[slice_idx]
        positions = get_patch_positions(h, w, patch_size, stride)
        
        # Extract patches with safe boundary handling
        patches = [extract_patch_safe(slice_array, x, y, patch_size) for x, y in positions]
        norm_patches = [normalize_patch(p) for p in patches]
        
        # Batch processing
        all_results = []
        for i in range(0, len(norm_patches), batch_size):
            batch = norm_patches[i:i+batch_size]
            batch_results = model(batch, verbose=False)
            all_results.extend(batch_results)
        
        # Collect all detections (before filtering)
        boxes_all = []
        scores_all = []
        
        for (x_off, y_off), res in zip(positions, all_results):
            if res.boxes is None:
                continue
            
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            
            for (x1, y1, x2, y2), score in zip(boxes, scores):
                if score >= conf_thresh:
                    boxes_all.append([x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off])
                    scores_all.append(score)
        
        # Apply NMS
        if boxes_all:
            boxes_tensor = torch.tensor(boxes_all, dtype=torch.float32)
            scores_tensor = torch.tensor(scores_all, dtype=torch.float32)
            nms_indices = nms(boxes_tensor, scores_tensor, nms_thresh)
            nms_indices = nms_indices.cpu().numpy()
            
            nms_boxes = np.array([boxes_all[i] for i in nms_indices])
            nms_scores = np.array([scores_all[i] for i in nms_indices])
        else:
            nms_boxes = np.array([]).reshape(0, 4)
            nms_scores = np.array([])
        
        # Apply size filtering
        if len(nms_boxes) > 0:
            size_filtered_boxes, size_filtered_scores, size_filter_mask = size_filter.filter_detections(
                nms_boxes, nms_scores
            )
        else:
            size_filtered_boxes = nms_boxes
            size_filtered_scores = nms_scores
        
        # Apply intensity filtering
        if len(size_filtered_boxes) > 0:
            intensity_filtered_boxes, intensity_filtered_scores, intensity_filter_mask, intensity_features = intensity_filter.filter_detections(
                size_filtered_boxes, size_filtered_scores, slice_array
            )
        else:
            intensity_filtered_boxes = size_filtered_boxes
            intensity_filtered_scores = size_filtered_scores
            intensity_features = []
        
        # Store detections for plane coverage analysis
        plane_filter.store_slice_detections(
            slice_idx, intensity_filtered_boxes, intensity_filtered_scores, intensity_features
        )
        
        # Store slice statistics
        slice_stats = {
            'slice_idx': int(slice_idx),
            'raw_detections': int(len(boxes_all)),
            'post_nms_detections': int(len(nms_boxes)),
            'post_size_filter_detections': int(len(size_filtered_boxes)),
            'post_intensity_filter_detections': int(len(intensity_filtered_boxes)),
            'size_filter_removed': int(len(nms_boxes) - len(size_filtered_boxes)),
            'intensity_filter_removed': int(len(size_filtered_boxes) - len(intensity_filtered_boxes))
        }
        slice_statistics.append(slice_stats)
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # PHASE 2: Apply 3D plane coverage filtering
    print("\n=== PHASE 2: 3D plane coverage analysis ===")
    
    filtered_detections_per_slice, track_analysis = plane_filter.apply_plane_coverage_filtering(n_slices)
    
    # PHASE 3: Generate output visualization
    print("\n=== PHASE 3: Generating output visualization ===")
    
    output_stack = []
    
    # Add color legend
    legend_slice = create_color_legend(width=w, height=h)
    output_stack.append(legend_slice)
    
    # Update slice statistics with plane coverage results
    for slice_idx in range(n_slices):
        final_detections = filtered_detections_per_slice[slice_idx]
        n_final_detections = len(final_detections['boxes'])
        
        # Update statistics
        slice_statistics[slice_idx]['post_plane_coverage_detections'] = int(n_final_detections)
        slice_statistics[slice_idx]['plane_coverage_removed'] = int(
            slice_statistics[slice_idx]['post_intensity_filter_detections'] - n_final_detections
        )
        
        # Create RGB output with final filtered detections
        slice_array = tif_stack[slice_idx]
        rgb = normalize_patch(slice_array)
        
        # Draw final boxes with confidence-based colors
        for i, (box, conf) in enumerate(zip(final_detections['boxes'], final_detections['scores'])):
            x1, y1, x2, y2 = map(int, box)
            color = confidence_to_color(conf)
            cv2.rectangle(rgb, (x1, y1), (x2, y2), color, thickness=1)
            
            # Optionally draw track ID
            if i < len(final_detections['track_ids']):
                track_id = final_detections['track_ids'][i]
                cv2.putText(rgb, f'T{track_id}', (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Add detection counts to image
        cv2.putText(rgb, f'Final: {n_final_detections}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(rgb, f'Pre-3D: {slice_statistics[slice_idx]["post_intensity_filter_detections"]}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(rgb, f'Raw: {slice_statistics[slice_idx]["raw_detections"]}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        output_stack.append(rgb)
    
    # Compile comprehensive statistics
    overall_stats = {
        'processing_info': {
            'timestamp': datetime.now().isoformat(),
            'tif_path': tif_path,
            'model_path': model_path,
            'patch_size': patch_size,
            'stride': stride,
            'conf_thresh': conf_thresh,
            'nms_thresh': nms_thresh,
            'size_filtering_mode': size_filtering_mode,
            'intensity_filtering_mode': intensity_filtering_mode,
            'plane_coverage_mode': plane_coverage_mode,
            'n_slices': n_slices
        },
        'size_filter_stats': size_filter.get_filtering_stats() if hasattr(size_filter, 'get_filtering_stats') else {},
        'intensity_filter_stats': intensity_filter.get_filtering_stats() if hasattr(intensity_filter, 'get_filtering_stats') else {},
        'plane_coverage_stats': plane_filter.get_filtering_stats(),
        'track_analysis': track_analysis,
        'slice_statistics': slice_statistics,
        'summary': {
            'total_raw_detections': int(sum(s['raw_detections'] for s in slice_statistics)),
            'total_post_nms': int(sum(s['post_nms_detections'] for s in slice_statistics)),
            'total_post_size_filter': int(sum(s['post_size_filter_detections'] for s in slice_statistics)),
            'total_post_intensity_filter': int(sum(s['post_intensity_filter_detections'] for s in slice_statistics)),
            'total_final_detections': int(sum(s['post_plane_coverage_detections'] for s in slice_statistics)),
            'total_removed_by_size_filter': int(sum(s['size_filter_removed'] for s in slice_statistics)),
            'total_removed_by_intensity_filter': int(sum(s['intensity_filter_removed'] for s in slice_statistics)),
            'total_removed_by_plane_coverage': int(sum(s['plane_coverage_removed'] for s in slice_statistics))
        }
    }
    
    # Save results
    tifffile.imwrite(save_path, np.stack(output_stack), photometric='rgb')
    print(f"Saved comprehensive filtered RGB stack to {save_path}")
    
    # Save statistics
    if save_stats:
        # Convert all numpy types to native Python types for JSON serialization
        overall_stats_json = convert_numpy_types(overall_stats)
        
        stats_path = save_path.replace('.tif', '_comprehensive_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(overall_stats_json, f, indent=2)
        print(f"Saved comprehensive filtering statistics to {stats_path}")
    
    # Print comprehensive summary
    summary = overall_stats['summary']
    print(f"\n=== COMPREHENSIVE FILTERING SUMMARY ===")
    print(f"Total raw detections: {summary['total_raw_detections']}")
    print(f"After NMS: {summary['total_post_nms']}")
    print(f"After size filtering: {summary['total_post_size_filter']} "
          f"(removed {summary['total_removed_by_size_filter']})")
    print(f"After intensity filtering: {summary['total_post_intensity_filter']} "
          f"(removed {summary['total_removed_by_intensity_filter']})")
    print(f"After plane coverage filtering: {summary['total_final_detections']} "
          f"(removed {summary['total_removed_by_plane_coverage']})")
    
    if summary['total_post_nms'] > 0:
        total_removed = (summary['total_removed_by_size_filter'] + 
                        summary['total_removed_by_intensity_filter'] + 
                        summary['total_removed_by_plane_coverage'])
        overall_removal_rate = total_removed / summary['total_post_nms'] * 100
        print(f"Overall filtering efficiency: {overall_removal_rate:.1f}% of post-NMS detections removed")
    
    # Print 3D analysis results
    print(f"\n=== 3D TRACK ANALYSIS ===")
    track_stats = track_analysis['filtering_summary']
    print(f"Total tracks built: {track_analysis['total_tracks']}")
    print(f"Valid tracks: {track_analysis['valid_tracks']}")
    print(f"Filtered tracks: {track_analysis['filtered_tracks']}")
    if 'pass_rate' in track_stats:
        print(f"Track pass rate: {track_stats['pass_rate']:.1f}%")
        print(f"  Filtered by plane coverage: {track_stats.get('filter_rate_insufficient_plane_coverage', 0):.1f}%")
        print(f"  Filtered by continuity: {track_stats.get('filter_rate_poor_continuity', 0):.1f}%")
        print(f"  Filtered by gaps: {track_stats.get('filter_rate_excessive_gaps', 0):.1f}%")
        print(f"  Filtered by track length: {track_stats.get('filter_rate_track_too_short', 0):.1f}%")
    
    return overall_stats

# Usage example
if __name__ == "__main__":
    # Replace with your paths
    tif_path= "dataset/DeepD3_Benchmark.tif"
    model_path= r"uncertainty_enhanced_yolo/enhanced_dataset/models/stage2_main_training/weights/best.pt"
    best_path = r"runs/segment/train18/weights/best.pt"
    output_dir = r"yolo_optimized_hybrid_bb/prob_unet_enhanced2"
    d3set_path = r"dataset/DeepD3_Validation.d3set"
    
    # Run comprehensive filtering pipeline
    stats = run_yolo_with_comprehensive_filtering(
        tif_path=tif_path,
        model_path=model_path,
        patch_size=192,
        stride=96,  # 50% overlap
        conf_thresh=0.25,
        nms_thresh=0.5,
        batch_size=8,
        size_filtering_mode='conservative',      # 'conservative' or 'aggressive'
        intensity_filtering_mode='moderate',     # 'strict', 'moderate', or 'loose'
        plane_coverage_mode='moderate',          # 'strict', 'moderate', or 'loose'
        save_path="comprehensive_filtered_output.tif",
        save_stats=False
    )
    
    # You can also use custom plane coverage thresholds
    # custom_plane_thresholds = {
    #     'min_plane_coverage': 2,           # Must appear in at least 2 slices
    #     'max_spatial_distance': 18,        # Max 18-pixel movement between slices
    #     'min_continuity_ratio': 0.5,       # 50% continuity required
    #     'max_gap_size': 1,                  # Allow 1-slice gaps
    #     'edge_slice_tolerance': 3,          # Edge tolerance
    #     'min_track_length': 2,              # Minimum track length
    #     'spatial_consistency_weight': 0.25  # Spatial consistency weight
    # }
    # 
    # stats_custom = run_yolo_with_comprehensive_filtering(
    #     tif_path=tif_path,
    #     model_path=model_path,
    #     size_filtering_mode='conservative',
    #     intensity_filtering_mode='moderate',
    #     plane_coverage_mode=custom_plane_thresholds,
    #     save_path="custom_plane_filtered_output.tif"
    # )