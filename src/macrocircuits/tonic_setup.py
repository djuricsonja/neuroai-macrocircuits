"""On-demand installation of the tonic RL library.

tonic isn't on PyPI and its setup.py can't be pip-installed (flat layout with a
stray top-level `data` package), so it is cloned and imported from the checkout.
"""

import subprocess
import sys
from pathlib import Path

TONIC_URL = 'https://github.com/neuromatch/tonic'


def ensure_tonic(directory=None):
    """Clones tonic if needed and puts it on the import path.

    Parameters:
    - directory (os.PathLike, optional): Where the checkout lives. Defaults to
      `tonic/` under the current working directory, i.e. next to the notebook.

    Returns:
    Path: the directory tonic was imported from.
    """
    directory = Path(directory) if directory is not None else Path.cwd() / 'tonic'

    if not directory.exists():
        subprocess.run(
            ['git', 'clone', '--quiet', '--depth', '1', TONIC_URL, str(directory)],
            check=True,
        )

    # Import from the checkout rather than `%cd tonic`, which would move the working
    # directory and break the relative paths (output_videos/, data/) that display_video
    # and the training cells rely on.
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

    return directory
