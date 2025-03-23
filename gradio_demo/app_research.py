# This Gradio demo code is from https://github.com/cvlab-kaist/locotrack/blob/main/demo/demo.py 
# We updated it to work with CoTracker3 models. We thank authors of LocoTrack
# for such an amazing Gradio demo.

import os
import sys
import uuid
import json

import gradio as gr
import mediapy
import numpy as np
import cv2
import matplotlib
import torch
import colorsys
import random
from typing import List, Optional, Sequence, Tuple
import math

import numpy as np
import rerun as rr

# Generate random colormaps for visualizing different points.
def get_colors(num_colors: int) -> List[Tuple[int, int, int]]:
  """Gets colormap for points."""
  colors = []
  for i in np.arange(0.0, 360.0, 360.0 / num_colors):
    hue = i / 360.0
    lightness = (50 + np.random.rand() * 10) / 100.0
    saturation = (90 + np.random.rand() * 10) / 100.0
    color = colorsys.hls_to_rgb(hue, lightness, saturation)
    colors.append(
        (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
    )
  random.shuffle(colors)
  return colors

def get_points_on_a_grid(
    size: int,
    extent: Tuple[float, ...],
    center: Optional[Tuple[float, ...]] = None,
    device: Optional[torch.device] = torch.device("cpu"),
):
    r"""Get a grid of points covering a rectangular region

    `get_points_on_a_grid(size, extent)` generates a :attr:`size` by
    :attr:`size` grid fo points distributed to cover a rectangular area
    specified by `extent`.

    The `extent` is a pair of integer :math:`(H,W)` specifying the height
    and width of the rectangle.

    Optionally, the :attr:`center` can be specified as a pair :math:`(c_y,c_x)`
    specifying the vertical and horizontal center coordinates. The center
    defaults to the middle of the extent.

    Points are distributed uniformly within the rectangle leaving a margin
    :math:`m=W/64` from the border.

    It returns a :math:`(1, \text{size} \times \text{size}, 2)` tensor of
    points :math:`P_{ij}=(x_i, y_i)` where

    .. math::
        P_{ij} = \left(
             c_x + m -\frac{W}{2} + \frac{W - 2m}{\text{size} - 1}\, j,~
             c_y + m -\frac{H}{2} + \frac{H - 2m}{\text{size} - 1}\, i
        \right)

    Points are returned in row-major order.

    Args:
        size (int): grid size.
        extent (tuple): height and with of the grid extent.
        center (tuple, optional): grid center.
        device (str, optional): Defaults to `"cpu"`.

    Returns:
        Tensor: grid.
    """
    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]

    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)

def log_video(frames: np.ndarray, name_visualization: str) -> None:
    """Log video frames to rerun."""
    for i, frame in enumerate(frames):
        rr.set_time_sequence("frameid", i)
        rr.log(name_visualization, rr.Image(frame))

def log_trajectory(
    query_frame: np.ndarray, query_xys: np.ndarray, colors: Optional[np.ndarray] = None
) -> None:
    """Log query image and points to rerun."""
    rr.set_time_sequence("frameid", 0)
    rr.log("query_frame", rr.Image(query_frame))
    rr.log("query_frame/query_points", rr.Points2D(query_xys, colors=colors))

def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculate angle between three points."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def log_angles(
    frame_id: int,
    point_tracks: np.ndarray,
    visibles: np.ndarray,
    angle_triplets: List[List[int]] = [[1, 2, 3], [3, 4, 5]]  # Example triplets to track
) -> None:
    """Log angles between specified triplets of points.
    
    Args:
        frame_id: Current frame number
        point_tracks: Array of point tracks
        visibles: Array indicating point visibility
        angle_triplets: List of triplets specifying which points to calculate angles between.
                       Each triplet is [p1, p2, p3] where p2 is the middle point.
    """
    num_points = point_tracks.shape[0]
    
    # Track angles for each specified triplet
    for triplet_idx, (p1_idx, p2_idx, p3_idx) in enumerate(angle_triplets):
        # Check indices are valid
        if max(p1_idx, p2_idx, p3_idx) >= num_points:
            continue
            
        # Check all points in triplet are visible
        if visibles[p1_idx, frame_id] and visibles[p2_idx, frame_id] and visibles[p3_idx, frame_id]:
            p1 = point_tracks[p1_idx, frame_id]
            p2 = point_tracks[p2_idx, frame_id] 
            p3 = point_tracks[p3_idx, frame_id]
            
            angle = calculate_angle(p1, p2, p3)
            
            # Log angle for this specific triplet
            rr.set_time_sequence("frameid", frame_id)
            rr.log(f"angles_{triplet_idx}/angle", rr.Scalar(angle))

def log_tracks(frame_points_sequence: List[List[dict]], image_size: Tuple[int, int]) -> None:
    """Log individual point trajectories and connections to Rerun.
    
    Args:
        frame_points_sequence: List of frames, where each frame is a list of point dictionaries
        image_size: Tuple of (height, width) for creating visualization
    """
    height, width = image_size

    # Create black images for different visualizations
    all_points_image = np.zeros((height, width, 3), dtype=np.uint8)
    connections_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Create individual point track images
    point_track_images = [
        np.zeros((height, width, 3), dtype=np.uint8),
        np.zeros((height, width, 3), dtype=np.uint8),
        np.zeros((height, width, 3), dtype=np.uint8),
        np.zeros((height, width, 3), dtype=np.uint8),
        np.zeros((height, width, 3), dtype=np.uint8),
        np.zeros((height, width, 3), dtype=np.uint8)
    ]
    # Process each frame
    for frame_id, frame_points in enumerate(frame_points_sequence):
        # Draw each visible point and its trajectory
        for i, point in enumerate(frame_points):
            if point['visible']:
                x = int(point['x'])
                y = int(point['y'])
                            
                # Draw current point as circle
                cv2.circle(point_track_images[i], (x, y), 5, (255, 255, 255), -1)
                cv2.circle(all_points_image, (x, y), 5, (255, 255, 255), -1)
                                       
                # Log individual point image
                rr.set_time_sequence("frameid", frame_id)
                rr.log(f"points/point_{NAME_POINTS[i]}", rr.Image(point_track_images[i]))
                rr.log("points/all_points", rr.Image(all_points_image))

        # Calculate and log angles between points 1,2,3
        if len(frame_points) > 3 and all(frame_points[i]['visible'] for i in [1,2,3]):
            p1 = np.array([frame_points[1]['x'], frame_points[1]['y']])
            p2 = np.array([frame_points[2]['x'], frame_points[2]['y']]) 
            p3 = np.array([frame_points[3]['x'], frame_points[3]['y']])
            
            # Calculate vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle in degrees
            angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
            
            # Log angle to Rerun
            rr.set_time_sequence("frameid", frame_id)
            rr.log("angles/angle", rr.Scalar(angle))
    
        # Draw connections between consecutive points
        for i in range(len(frame_points)-1):
            
            # Create robot visualization with lines
            robot_image = np.zeros((height, width, 3), dtype=np.uint8)
               
            # Draw line from point 0 to 2 if both visible
            if len(frame_points) > 2 and frame_points[0]['visible'] and frame_points[2]['visible']:
                pt1 = (int(frame_points[0]['x']), int(frame_points[0]['y']))
                pt2 = (int(frame_points[2]['x']), int(frame_points[2]['y']))
                cv2.line(robot_image, pt1, pt2, (255, 255, 255), 10)
            
            # Draw line from point 2 to 4 if both visible
            if len(frame_points) > 4 and frame_points[2]['visible'] and frame_points[4]['visible']:
                pt1 = (int(frame_points[2]['x']), int(frame_points[2]['y']))
                pt2 = (int(frame_points[4]['x']), int(frame_points[4]['y']))
                cv2.line(robot_image, pt1, pt2, (0, 255, 0), 5)

            # Only connect consecutive points in sequence
            # Skip if either point is not visible
            if not frame_points[i]['visible'] or not frame_points[i+1]['visible']:
                continue
            
            pt1 = (int(frame_points[i]['x']), int(frame_points[i]['y']))
            pt2 = (int(frame_points[i+1]['x']), int(frame_points[i+1]['y']))
            # Draw line between points
            cv2.line(connections_image, pt1, pt2, (255, 255, 255), 2)
        
            # Log to Rerun
            rr.set_time_sequence("frameid", frame_id)
            rr.log("points/connections", rr.Image(connections_image))
            rr.log("points/robot", rr.Image(robot_image))
            
def paint_point_track_visualizate(
    frames: np.ndarray,
    point_tracks: np.ndarray, 
    visibles: np.ndarray,
    colormap: Optional[List[Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """Converts a sequence of points to color code video with progressive tracking logs.

    Args:
        frames: [num_frames, height, width, 3], np.uint8, [0, 255]
        point_tracks: [num_points, num_frames, 2], np.float32, [0, width / height]
        visibles: [num_points, num_frames], bool.
        colormap: colormap for points, each point has a different RGB color.

    Returns:
        video: [num_frames, height, width, 3], np.uint8, [0, 255]
    """
    try:
        num_points, num_frames = point_tracks.shape[0:2]
        if colormap is None:
            colormap = get_colors(num_colors=num_points)
        height, width = frames.shape[1:3]
        dot_size_as_fraction_of_min_edge = 0.015
        radius = int(round(min(height, width) * dot_size_as_fraction_of_min_edge))
        diam = radius * 2 + 1
        quadratic_y = np.square(np.arange(diam)[:, np.newaxis] - radius - 1)
        quadratic_x = np.square(np.arange(diam)[np.newaxis, :] - radius - 1)
        icon = (quadratic_y + quadratic_x) - (radius**2) / 2.0
        sharpness = 0.15
        icon = np.clip(icon / (radius * 2 * sharpness), 0, 1)
        icon = 1 - icon[:, :, np.newaxis]
        icon1 = np.pad(icon, [(0, 1), (0, 1), (0, 0)])
        icon2 = np.pad(icon, [(1, 0), (0, 1), (0, 0)])
        icon3 = np.pad(icon, [(0, 1), (1, 0), (0, 0)])
        icon4 = np.pad(icon, [(1, 0), (1, 0), (0, 0)])

        video = frames.copy()

        log_video(frames, "original")

        frame_points_sequence = []
        show_past_points = True  
        for t in range(num_frames):
            image = np.pad(
                video[t],
                [(radius + 1, radius + 1), (radius + 1, radius + 1), (0, 0)],
            )
            
            # Track which colors have been used this frame
            used_colors = set()
            
            # Store points for this frame
            frame_points = []
            
            # Process points in reverse order
            for i in range(num_points-1, -1, -1):
                x, y = point_tracks[i, t, :] + 0.5
                x = min(max(x, 0.0), width)
                y = min(max(y, 0.0), height)

                if visibles[i, t]:
                    # Get color for this point
                    point_color = tuple(colormap[i])
                    
                    # If color already used this frame, make point invisible
                    if point_color in used_colors:
                        visibles[i, t] = False
                        continue
                        
                    used_colors.add(point_color)
                    
                    # Store point data
                    frame_points.append({
                        'x': x,
                        'y': y,
                        'color': point_color,
                        'visible': True,
                        'point_id': str(colormap[i])
                    })
                    
                    x1, y1 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
                    x2, y2 = x1 + 1, y1 + 1
                    patch = (
                        icon1 * (x2 - x) * (y2 - y)
                        + icon2 * (x2 - x) * (y - y1)
                        + icon3 * (x - x1) * (y2 - y)
                        + icon4 * (x - x1) * (y - y1)
                    )
                    x_ub = x1 + 2 * radius + 2
                    y_ub = y1 + 2 * radius + 2
                    image[y1:y_ub, x1:x_ub, :] = (1 - patch) * image[
                        y1:y_ub, x1:x_ub, :
                    ] + patch * np.array(colormap[i])[np.newaxis, np.newaxis, :]

            # Draw past points if the flag is set
            if show_past_points:
                for i in range(num_points):
                    for past_t in range(t):  # Show past points up to the current frame
                        if visibles[i, past_t]:
                            past_x, past_y = point_tracks[i, past_t, :] + 0.5
                            past_x = min(max(past_x, 0.0), width)
                            past_y = min(max(past_y, 0.0), height)
                            # Use the same patch idea instead of cv2.circle
                            past_x1, past_y1 = np.floor(past_x).astype(np.int32), np.floor(past_y).astype(np.int32)
                            past_x2, past_y2 = past_x1 + 1, past_y1 + 1
                            past_patch = (
                                icon1 * (past_x2 - past_x) * (past_y2 - past_y)
                                + icon2 * (past_x2 - past_x) * (past_y - past_y1)
                                + icon3 * (past_x - past_x1) * (past_y2 - past_y)
                                + icon4 * (past_x - past_x1) * (past_y - past_y1)
                            )
                            past_x_ub = past_x1 + 2 * radius + 2
                            past_y_ub = past_y1 + 2 * radius + 2
                            image[past_y1:past_y_ub, past_x1:past_x_ub, :] = (1 - past_patch) * image[
                                past_y1:past_y_ub, past_x1:past_x_ub, :
                            ] + past_patch * np.array(colormap[i])[np.newaxis, np.newaxis, :]

            # Store frame points for later logging
            frame_points_sequence.append(frame_points)
                
            video[t] = image[
                radius + 1 : -radius - 1, radius + 1 :  -radius - 1
            ].astype(np.uint8)
        
        # Log progressive trajectories with visible points only
        log_tracks(frame_points_sequence, (height, width))

        log_video(video, "tracking")
    except Exception as e:
        print("Error in the tracker visualization")
        print(e)
            
    except Exception as e:
        print("Error in the tracker visualization")
        print(e)
    
    return video

PREVIEW_WIDTH = 768 # Width of the preview video
VIDEO_INPUT_RESO = (384, 512) # Resolution of the input video
POINT_SIZE = 4 # Size of the query point in the preview video
FRAME_LIMIT = 3000 # Limit the number of frames to process
NAME_POINTS={
    0:"head_mid_top",
    1:"torso_mid_back",
    2:"tail_top_back",
    3:"tail_mid_back",
    4:"tail_bottom_back"
}
COLOR_POINTS={
    0:(255,0,0),
    1:(0,255,0),
    2:(0,0,255),
    3:(255,255,0),
    4:(0,255,255)
}


def get_point(frame_num, video_queried_preview, query_points, query_points_color, query_count, evt: gr.SelectData):
    print(f"You selected {(evt.index[0], evt.index[1], frame_num)}")

    current_frame = video_queried_preview[int(frame_num)]

    # Check if we already have 5 points
    if len(query_points[int(frame_num)]) >= 5:
        gr.Info("You can only add 5 points per frame")
        return (
            current_frame,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count,
        )

    # Get the mouse click
    query_points[int(frame_num)].append((evt.index[0], evt.index[1], frame_num))

    # Choose the color for the point from the dictionary
    color = COLOR_POINTS[len(query_points[int(frame_num)])-1]

    # Choose the name for the point from the dictionary
    query_points_color[int(frame_num)].append(color)

    # Draw the point on the frame
    x, y = evt.index
    current_frame_draw = cv2.circle(current_frame, (x, y), POINT_SIZE, color, -1)

    # Update the frame
    video_queried_preview[int(frame_num)] = current_frame_draw

    # Update the query count
    query_count += 1

    return (
        current_frame_draw, # Updated frame for preview
        video_queried_preview, # Updated preview video
        query_points, # Updated query points
        query_points_color, # Updated query points color
        query_count # Updated query count
    )


def undo_point(frame_num, video_preview, video_queried_preview, query_points, query_points_color, query_count):
    if len(query_points[int(frame_num)]) == 0:
        return (
            video_queried_preview[int(frame_num)],
            video_queried_preview,
            query_points,
            query_points_color,
            query_count
        )

    # Get the last point
    query_points[int(frame_num)].pop(-1)
    query_points_color[int(frame_num)].pop(-1)

    # Redraw the frame
    current_frame_draw = video_preview[int(frame_num)].copy()
    for point, color in zip(query_points[int(frame_num)], query_points_color[int(frame_num)]):
        x, y, _ = point
        current_frame_draw = cv2.circle(current_frame_draw, (x, y), POINT_SIZE, color, -1)

    # Update the query count
    query_count -= 1

    # Update the frame
    video_queried_preview[int(frame_num)] = current_frame_draw
    return (
        current_frame_draw, # Updated frame for preview
        video_queried_preview, # Updated preview video
        query_points, # Updated query points
        query_points_color, # Updated query points color
        query_count # Updated query count
    )


def clear_frame_fn(frame_num, video_preview, video_queried_preview, query_points, query_points_color, query_count):
    query_count -= len(query_points[int(frame_num)])

    query_points[int(frame_num)] = []
    query_points_color[int(frame_num)] = []

    video_queried_preview[int(frame_num)] = video_preview[int(frame_num)].copy()

    return (
        video_preview[int(frame_num)], # Set the preview frame to the original frame
        video_queried_preview, 
        query_points, # Cleared query points
        query_points_color, # Cleared query points color
        query_count # New query count
    )



def clear_all_fn(frame_num, video_preview):
    return (
        video_preview[int(frame_num)],
        video_preview.copy(),
        [[] for _ in range(len(video_preview))],
        [[] for _ in range(len(video_preview))],
        0
    )


def choose_frame(frame_num, video_preview_array):
    return video_preview_array[int(frame_num)]


def preprocess_video_input(video_path):
    video_arr = mediapy.read_video(video_path)
    video_fps = video_arr.metadata.fps
    num_frames = video_arr.shape[0]
    if num_frames > FRAME_LIMIT:
        gr.Warning(f"The video is too long. Only the first {FRAME_LIMIT} frames will be used.", duration=5)
        video_arr = video_arr[:FRAME_LIMIT]
        num_frames = FRAME_LIMIT

    # Resize to preview size for faster processing, width = PREVIEW_WIDTH
    height, width = video_arr.shape[1:3]
    new_height, new_width = int(PREVIEW_WIDTH * height / width), PREVIEW_WIDTH

    preview_video = mediapy.resize_video(video_arr, (new_height, new_width))
    input_video = mediapy.resize_video(video_arr, VIDEO_INPUT_RESO)

    preview_video = np.array(preview_video)
    input_video = np.array(input_video)
    
    interactive = True

    return (
        video_arr, # Original video
        preview_video, # Original preview video, resized for faster processing
        preview_video.copy(), # Copy of preview video for visualization
        input_video, # Resized video input for model
        # None, # video_feature, # Extracted feature
        video_fps, # Set the video FPS
        gr.update(open=False), # Close the video input drawer
        # tracking_mode, # Set the tracking mode
        preview_video[0], # Set the preview frame to the first frame
        gr.update(minimum=0, maximum=num_frames - 1, value=0, interactive=interactive), # Set slider interactive
        [[] for _ in range(num_frames)], # Set query_points to empty
        [[] for _ in range(num_frames)], # Set query_points_color to empty
        [[] for _ in range(num_frames)], 
        0, # Set query count to 0
        gr.update(interactive=interactive), # Make the buttons interactive
        gr.update(interactive=interactive),
        gr.update(interactive=interactive),
        gr.update(interactive=True),
    )


def track(
    video_preview,
    video_input, 
    video_fps, 
    query_points, 
    query_points_color, 
    query_count, 
):
    # Save tracking points data
    points_data = {
        "metadata": {
            "fps": video_fps,
            "point_names": NAME_POINTS
        },
        "frames": []
    }
    
    # Convert query points to structured data
    for frame_idx, frame_points in enumerate(query_points):
        if frame_points:  # If frame has points
            frame_data = {
                "frame_number": frame_idx,
                "points": []
            }
            for point_idx, (x, y, _) in enumerate(frame_points):
                point_data = {
                    "name": NAME_POINTS[point_idx],
                    "x": float(x),
                    "y": float(y),
                    "color": query_points_color[frame_idx][point_idx]
                }
                frame_data["points"].append(point_data)
            points_data["frames"].append(frame_data)
    
    # Save to JSON file
    output_dir = os.path.join(os.path.dirname(__file__), "points_data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "tracking_points.json")
    
    with open(output_file, "w") as f:
        json.dump(points_data, f, indent=2)
    
    print(f"Saved tracking points data to: {output_file}")
    print("\nExample of saved data structure:")
    print(json.dumps(points_data, indent=2))

    tracking_mode = 'selected'
    if query_count == 0: 
        tracking_mode='grid'
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float if device == "cuda" else torch.float

    # Convert query points to tensor, normalize to input resolution
    if tracking_mode!='grid':
        query_points_tensor = []
        for frame_points in query_points:
            query_points_tensor.extend(frame_points)
        
        query_points_tensor = torch.tensor(query_points_tensor).float()
        query_points_tensor *= torch.tensor([
            VIDEO_INPUT_RESO[1], VIDEO_INPUT_RESO[0], 1
        ]) / torch.tensor([
            [video_preview.shape[2], video_preview.shape[1], 1]
        ])
        query_points_tensor = query_points_tensor[None].flip(-1).to(device, dtype) # xyt -> tyx
        query_points_tensor = query_points_tensor[:, :, [0, 2, 1]] # tyx -> txy

    video_input = torch.tensor(video_input).unsqueeze(0).to(device, dtype)

    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    model = model.to(device)

    video_input = video_input.permute(0, 1, 4, 2, 3)
    if tracking_mode=='grid':
        xy = get_points_on_a_grid(15, video_input.shape[3:], device=device)
        queries = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2).to(device)  #
        add_support_grid=False
        cmap = matplotlib.colormaps.get_cmap("gist_rainbow")
        query_points_color = [[]]
        query_count = queries.shape[1]
        for i in range(query_count):
            # Choose the color for the point from matplotlib colormap
            color = cmap(i / float(query_count))
            color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            query_points_color[0].append(color)

    else:
        queries = query_points_tensor
        add_support_grid=True

    model(video_chunk=video_input, is_first_step=True, grid_size=0, queries=queries, add_support_grid=add_support_grid)
    
    for ind in range(0, video_input.shape[1] - model.step, model.step):
        pred_tracks, pred_visibility = model(
            video_chunk=video_input[:, ind : ind + model.step * 2],
            grid_size=0, 
            queries=queries, 
            add_support_grid=add_support_grid
        )  # B T N 2,  B T N 1
    tracks = (pred_tracks * torch.tensor([video_preview.shape[2], video_preview.shape[1]]).to(device) / torch.tensor([VIDEO_INPUT_RESO[1], VIDEO_INPUT_RESO[0]]).to(device))[0].permute(1, 0, 2).cpu().numpy()
    pred_occ = pred_visibility[0].permute(1, 0).cpu().numpy()

    # make color array
    colors = []
    for frame_colors in query_points_color:
        colors.extend(frame_colors)
    colors = np.array(colors)
    
    painted_video = paint_point_track_visualizate(video_preview,tracks,pred_occ,colors)

    # save video
    video_file_name = uuid.uuid4().hex + ".mp4"
    video_path = os.path.join(os.path.dirname(__file__), "tmp")
    video_file_path = os.path.join(video_path, video_file_name)
    os.makedirs(video_path, exist_ok=True)

    mediapy.write_video(video_file_path, painted_video, fps=video_fps)

    return video_file_path

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # State variables
    video = gr.State()
    video_queried_preview = gr.State() 
    video_preview = gr.State()
    video_input = gr.State()
    video_fps = gr.State(24)

    query_points = gr.State([])
    query_points_color = gr.State([])
    is_tracked_query = gr.State([])
    query_count = gr.State(0)

    # Title and description
    gr.Markdown(
        """
        # 🐊 Kinematic Adaptation in Space Robotics
        ## Inspired by Animal Righting Reflexes, using CoTracker3 and Rerun
        """
    )
    gr.Markdown(
        """
        ### First step
        Upload your video or select an example video, then click submit.
        """
    )

    # Video input section
    with gr.Row():
        with gr.Accordion("Your video input", open=True) as video_in_drawer:
            video_in = gr.Video(
                label="Video Input",
                format="mp4",
                elem_classes="video-input"
            )
            submit = gr.Button("Submit", variant="primary", scale=0)

            # Example videos
            lizard_1 = os.path.join(os.path.dirname(__file__), "videos", "video_1.mp4")
            lizard_2 = os.path.join(os.path.dirname(__file__), "videos", "video_2.mp4")
            lizard_3 = os.path.join(os.path.dirname(__file__), "videos", "video_3.mp4")
            lizard_4 = os.path.join(os.path.dirname(__file__), "videos", "video_4.mp4")
            lizard_5 = os.path.join(os.path.dirname(__file__), "videos", "video_5.mp4")

            gr.Examples(
                examples=[lizard_1, lizard_2, lizard_3, lizard_4, lizard_5],
                inputs=[video_in],
                label="Example Videos"
            )

    # Main interface section
    gr.Markdown(
        """
        ### Second step
        Please select 5 key points on the lizard in the following order:
        1. Head mid top
        2. Torso mid back 
        3. Tail top back
        4. Tail mid back
        5. Tail bottom back
        
        Then click "Track" to track these points through the video.
        """
    )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                query_frames = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    label="Choose Frame",
                    interactive=False
                )
            
            with gr.Row():
                undo = gr.Button("Undo", variant="secondary", interactive=False)
                clear_frame = gr.Button("Clear Frame", variant="secondary", interactive=False)
                clear_all = gr.Button("Clear All", variant="secondary", interactive=False)

            with gr.Row():
                current_frame = gr.Image(
                    label="Click to add query points",
                    type="numpy",
                    interactive=False,
                    elem_classes="frame-preview"
                )

            with gr.Row():
                track_button = gr.Button("Track", variant="primary", size="lg", interactive=False)

        # Right column - Output
        with gr.Column():
            output_video = gr.Video(
                label="Output Video",
                interactive=False,
                autoplay=True,
                loop=True,
                elem_classes="output-video"
            )

    # Initialize visualization
    rr.init("CoTracker Visualization Lizards v7")
    rr.spawn()

    # Event handlers
    submit.click(
        fn=preprocess_video_input,
        inputs=[video_in],
        outputs=[
            video, video_preview, video_queried_preview, video_input,
            video_fps, video_in_drawer, current_frame, query_frames,
            query_points, query_points_color, is_tracked_query,
            query_count, undo, clear_frame, clear_all, track_button
        ],
        queue=False
    )

    query_frames.change(
        fn=choose_frame,
        inputs=[query_frames, video_queried_preview],
        outputs=[current_frame],
        queue=False
    )

    current_frame.select(
        fn=get_point,
        inputs=[
            query_frames, video_queried_preview, query_points,
            query_points_color, query_count
        ],
        outputs=[
            current_frame, video_queried_preview, query_points,
            query_points_color, query_count
        ],
        queue=False
    )

    undo.click(
        fn=undo_point,
        inputs=[
            query_frames, video_preview, video_queried_preview,
            query_points, query_points_color, query_count
        ],
        outputs=[
            current_frame, video_queried_preview, query_points,
            query_points_color, query_count
        ],
        queue=False
    )

    clear_frame.click(
        fn=clear_frame_fn,
        inputs=[
            query_frames, video_preview, video_queried_preview,
            query_points, query_points_color, query_count
        ],
        outputs=[
            current_frame, video_queried_preview, query_points,
            query_points_color, query_count
        ],
        queue=False
    )

    clear_all.click(
        fn=clear_all_fn,
        inputs=[query_frames, video_preview],
        outputs=[
            current_frame, video_queried_preview, query_points,
            query_points_color, query_count
        ],
        queue=False
    )

    track_button.click(
        fn=track,
        inputs=[
            video_preview, video_input, video_fps,
            query_points, query_points_color, query_count
        ],
        outputs=[output_video],
        queue=True
    )

demo.launch(show_api=False, show_error=True, debug=True, share=True)