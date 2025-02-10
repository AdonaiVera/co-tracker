# This Gradio demo code is from https://github.com/cvlab-kaist/locotrack/blob/main/demo/demo.py 
# We updated it to work with CoTracker3 models. We thank authors of LocoTrack
# for such an amazing Gradio demo.

import os
import sys
import uuid

import gradio as gr
import mediapy
import numpy as np
import cv2
import matplotlib
import torch
import colorsys
import random
from typing import List, Optional, Sequence, Tuple

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

def log_tracks(
    frame_id: int,
    point_tracks: np.ndarray,
    visibles: np.ndarray,
    frame_size: Tuple[int, int]
) -> None:
    """Logs progressive trajectories of each key point on individual black images and a full trajectory image.

    Args:
        frame_id: The current frame index.
        point_tracks: [num_points, num_frames, 2], np.float32.
        visibles: [num_points, num_frames], bool.
        frame_size: (height, width) of the video frames.
    """
    height, width = frame_size
    num_points, num_frames = point_tracks.shape[:2]
    
    # Initialize a black image for the full trajectory
    full_trajectory_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(num_points):
        # Create a separate black image for each key point
        single_point_image = np.zeros((height, width, 3), dtype=np.uint8)
        trajectory = []

        for t in range(frame_id + 1):  # Track history up to the current frame
            if visibles[i, t]:
                x, y = point_tracks[i, t].astype(np.int32)
                trajectory.append((x, y))
        
        if trajectory:
            trajectory = np.array(trajectory)
            # Draw trajectory on the single-point image
            for j in range(len(trajectory) - 1):
                cv2.line(single_point_image, tuple(trajectory[j]), tuple(trajectory[j+1]), (255, 255, 255), 1)
                cv2.line(full_trajectory_image, tuple(trajectory[j]), tuple(trajectory[j+1]), (255, 255, 255), 1)
            
            # Log the individual key point trajectory
            rr.set_time_sequence("frameid", frame_id)
            rr.log(f"point_{i}/trajectory", rr.Image(single_point_image))
        
    # Log the full trajectory image
    rr.log("full_trajectory", rr.Image(full_trajectory_image))

def log_tracks_(
    tracks: np.ndarray,
    visibles: np.ndarray,
    colors: Optional[np.ndarray] = None,
    suffix="",
) -> None:
    """Log predicted point tracks to rerun."""
    num_frames = tracks.shape[1]

    for frame_id in range(num_frames):
        rr.set_time_sequence("frameid", frame_id)
        rr.log(
            "frame/points" + suffix,
            rr.Points2D(
                tracks[visibles[:, frame_id], frame_id],
                colors=colors[visibles[:, frame_id]],
            )
        )

        if frame_id == 0:
            continue

        visible_track_mask = visibles[:, frame_id - 1] * visibles[:, frame_id]
        rr.log(
            f"frame/tracks{suffix}",
            rr.LineStrips2D(
                tracks[visible_track_mask, frame_id - 1 : frame_id + 1],
                colors=colors[visible_track_mask],
            )
        )

def paint_point_track(
    frames: np.ndarray,
    point_tracks: np.ndarray,
    visibles: np.ndarray,
    colormap: Optional[List[Tuple[int, int, int]]] = None,
) -> np.ndarray:
  """Converts a sequence of points to color code video.

  Args:
    frames: [num_frames, height, width, 3], np.uint8, [0, 255]
    point_tracks: [num_points, num_frames, 2], np.float32, [0, width / height]
    visibles: [num_points, num_frames], bool
    colormap: colormap for points, each point has a different RGB color.

  Returns:
    video: [num_frames, height, width, 3], np.uint8, [0, 255]
  """
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
  for t in range(num_frames):
    # Pad so that points that extend outside the image frame don't crash us
    image = np.pad(
        video[t],
        [
            (radius + 1, radius + 1),
            (radius + 1, radius + 1),
            (0, 0),
        ],
    )
    for i in range(num_points):
      # The icon is centered at the center of a pixel, but the input coordinates
      # are raster coordinates.  Therefore, to render a point at (1,1) (which
      # lies on the corner between four pixels), we need 1/4 of the icon placed
      # centered on the 0'th row, 0'th column, etc.  We need to subtract
      # 0.5 to make the fractional position come out right.
      x, y = point_tracks[i, t, :] + 0.5
      x = min(max(x, 0.0), width)
      y = min(max(y, 0.0), height)

      if visibles[i, t]:
        x1, y1 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
        x2, y2 = x1 + 1, y1 + 1

        # bilinear interpolation
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

      # Remove the pad
      video[t] = image[
          radius + 1 : -radius - 1, radius + 1 : -radius - 1
      ].astype(np.uint8)
  return video

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
        
        for t in range(num_frames):
            image = np.pad(
                video[t],
                [(radius + 1, radius + 1), (radius + 1, radius + 1), (0, 0)],
            )
            for i in range(num_points):
                x, y = point_tracks[i, t, :] + 0.5
                x = min(max(x, 0.0), width)
                y = min(max(y, 0.0), height)

                if visibles[i, t]:
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
                    
                video[t] = image[
                    radius + 1 : -radius - 1, radius + 1 : -radius - 1
                ].astype(np.uint8)

            # Log progressive trajectories
            log_tracks(t, point_tracks, visibles, (height, width))

        log_video(video, "tracking")
            
    except Exception as e:
        print("Error in the tracker visualization")
        print(e)
    
    return video

PREVIEW_WIDTH = 768 # Width of the preview video
VIDEO_INPUT_RESO = (384, 512) # Resolution of the input video
POINT_SIZE = 4 # Size of the query point in the preview video
FRAME_LIMIT = 3000 # Limit the number of frames to process


def get_point(frame_num, video_queried_preview, query_points, query_points_color, query_count, evt: gr.SelectData):
    print(f"You selected {(evt.index[0], evt.index[1], frame_num)}")

    current_frame = video_queried_preview[int(frame_num)]

    # Get the mouse click
    query_points[int(frame_num)].append((evt.index[0], evt.index[1], frame_num))

    # Choose the color for the point from matplotlib colormap
    color = matplotlib.colormaps.get_cmap("gist_rainbow")(query_count % 20 / 20)
    color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
    # print(f"Color: {color}")
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


with gr.Blocks() as demo:
    video = gr.State()
    video_queried_preview = gr.State()
    video_preview = gr.State()
    video_input = gr.State()
    video_fps = gr.State(24)

    query_points = gr.State([])
    query_points_color = gr.State([])
    is_tracked_query = gr.State([])
    query_count = gr.State(0)

    gr.Markdown("# Kinematic Adaptation in Space Robotics Inspired by Animal Righting Reflexes, using CoTracker3 and Rerun 🐊 ")

    gr.Markdown("## First step: upload your video or select an example video, and click submit.")
    with gr.Row():
        with gr.Accordion("Your video input", open=True) as video_in_drawer:
            video_in = gr.Video(label="Video Input", format="mp4")
            submit = gr.Button("Submit", scale=0)

            import os
            lizard_1 = os.path.join(os.path.dirname(__file__), "videos", "video_1.mp4")
            lizard_2 = os.path.join(os.path.dirname(__file__), "videos", "video_2.mp4")
            lizard_3 = os.path.join(os.path.dirname(__file__), "videos", "video_3.mp4")
            lizard_4 = os.path.join(os.path.dirname(__file__), "videos", "video_4.mp4")
            lizard_5 = os.path.join(os.path.dirname(__file__), "videos", "video_5.mp4")

            gr.Examples(examples=[lizard_1, lizard_2, lizard_3, lizard_4, lizard_5], 
                        inputs = [
                            video_in
                        ],
                        )


    gr.Markdown("## Second step: Simply click \"Track\" to track a grid of points or select query points on the video before clicking")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                query_frames = gr.Slider(
                    minimum=0, maximum=100, value=0, step=1, label="Choose Frame", interactive=False)
            with gr.Row():
                undo = gr.Button("Undo", interactive=False)
                clear_frame = gr.Button("Clear Frame", interactive=False)
                clear_all = gr.Button("Clear All", interactive=False)

            with gr.Row():
                current_frame = gr.Image(
                    label="Click to add query points", 
                    type="numpy",
                    interactive=False
                )
            
            with gr.Row():
                track_button = gr.Button("Track", interactive=False)

        with gr.Column():
            output_video = gr.Video(
                label="Output Video",
                interactive=False,
                autoplay=True,
                loop=True,
            )

    rr.init("CoTracker Visualization Lizards")
    rr.spawn()

    submit.click(
        fn = preprocess_video_input, 
        inputs = [video_in], 
        outputs = [
            video,
            video_preview,
            video_queried_preview,
            video_input,
            video_fps,
            video_in_drawer,
            current_frame,
            query_frames,
            query_points,
            query_points_color,
            is_tracked_query,
            query_count,
            undo,
            clear_frame,
            clear_all,
            track_button,
        ],
        queue = False
    )

    query_frames.change(
        fn = choose_frame,
        inputs = [query_frames, video_queried_preview],
        outputs = [
            current_frame,
        ],
        queue = False
    )

    current_frame.select(
        fn = get_point, 
        inputs = [
            query_frames,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count,
        ], 
        outputs = [
            current_frame,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count
        ], 
        queue = False
    )
    
    undo.click(
        fn = undo_point,
        inputs = [
            query_frames,
            video_preview,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count
        ],
        outputs = [
            current_frame,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count
        ],
        queue = False
    )

    clear_frame.click(
        fn = clear_frame_fn,
        inputs = [
            query_frames,
            video_preview,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count
        ],
        outputs = [
            current_frame,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count
        ],
        queue = False
    )

    clear_all.click(
        fn = clear_all_fn,
        inputs = [
            query_frames,
            video_preview,
        ],
        outputs = [
            current_frame,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count
        ],
        queue = False
    )

    
    track_button.click(
        fn = track,
        inputs = [
            video_preview,
            video_input,
            video_fps,
            query_points,
            query_points_color,
            query_count,
        ],
        outputs = [
            output_video,
        ],
        queue = True,
    )

    
demo.launch(show_api=False, show_error=True, debug=True, share=True)