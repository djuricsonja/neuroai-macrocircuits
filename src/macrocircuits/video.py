"""Writing rollout frames to video files and displaying them in a notebook."""

import base64
import os
import typing as T

import imageio
import imageio_ffmpeg
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML

# matplotlib writes HTML5 video by shelling out to an `ffmpeg` executable, which is
# not on PATH. Point it at the binary that imageio-ffmpeg already ships.
matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()


def write_video(
    filepath: os.PathLike,
    frames: T.Iterable[np.ndarray],
    fps: int = 60,
    macro_block_size: T.Optional[int] = None,
    quality: int = 10,
    verbose: bool = False,
    **kwargs,
):
    """
    Saves a sequence of frames as a video file.

    Parameters:
    - filepath (os.PathLike): Path to save the video file.
    - frames (Iterable[np.ndarray]): An iterable of frames, where each frame is a numpy array.
    - fps (int, optional): Frames per second, defaults to 60.
    - macro_block_size (Optional[int], optional): Macro block size for video encoding, can affect compression efficiency.
    - quality (int, optional): Quality of the output video, higher values indicate better quality.
    - verbose (bool, optional): If True, prints the file path where the video is saved.
    - **kwargs: Additional keyword arguments passed to the imageio.get_writer function.

    Returns:
    None. The video is written to the specified filepath.
    """

    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with imageio.get_writer(
        filepath,
        fps=fps,
        macro_block_size=macro_block_size,
        quality=quality,
        **kwargs,
    ) as video:
        if verbose:
            print('Saving video to:', filepath)
        for frame in frames:
            video.append_data(frame)


def display_video(
    frames: T.Iterable[np.ndarray],
    filename='output_videos/temp.mp4',
    fps=60,
    **kwargs,
):
    """
    Displays a video within a Jupyter Notebook from an iterable of frames.

    Parameters:
    - frames (Iterable[np.ndarray]): An iterable of frames, where each frame is a numpy array.
    - filename (str, optional): Temporary filename to save the video before display, defaults to 'output_videos/temp.mp4'.
    - fps (int, optional): Frames per second for the video display, defaults to 60.
    - **kwargs: Additional keyword arguments passed to the write_video function.

    Returns:
    HTML object: An HTML video element that can be displayed in a Jupyter Notebook.
    """

    # Write video to a temporary file.
    filepath = os.path.abspath(filename)
    write_video(filepath, frames, fps=fps, verbose=False, **kwargs)

    # Embed the file just written directly as an HTML5 <video> tag, via base64. This
    # is what matplotlib's FuncAnimation.to_html5_video() also does under the hood --
    # but that path re-renders and re-encodes every frame a *second* time through
    # matplotlib's own animation writer first, which scales badly with frame count:
    # confirmed directly, a 2000-frame video's write_video call above finished in a
    # couple of minutes, then the matplotlib re-encode alone ran past 15 minutes
    # before a notebook execution timeout killed it -- for a video that had already
    # been sitting on disk, fully written, the whole time. Reading the bytes already
    # on disk skips that redundant pass entirely.
    height, width, _ = frames[0].shape
    with open(filepath, 'rb') as f:
        video_b64 = base64.b64encode(f.read()).decode('ascii')
    return HTML(
        f'<video width="{width}" height="{height}" controls autoplay>'
        f'<source src="data:video/mp4;base64,{video_b64}" type="video/mp4">'
        f'</video>'
    )
