"""Dataset helpers and compatibility settings for the local experiments.

MIKASA videos are decoded with LeRobot's PyAV fallback in our WSL setup. The
installed TorchCodec wheel cannot load because the matching FFmpeg shared
libraries are not present. LeRobot logs the failed capability probe at
WARNING level before falling back successfully; suppress only that noisy
probe logger so experiment output contains actual warnings and errors.
"""

import logging


logging.getLogger("lerobot.utils.import_utils").setLevel(logging.ERROR)
