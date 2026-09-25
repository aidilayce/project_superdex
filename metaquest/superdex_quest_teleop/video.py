# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MP4 encoding of rendered replay frames.

The viewer renders each video frame and sends it as a JPEG; this pipes the
JPEGs into ffmpeg (the binary bundled with ``imageio-ffmpeg``, or one on the
PATH) and encodes H.264 (``libx264``; MPEG-4 Part 2 if the ffmpeg build has
no x264).
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path


def find_ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - not installed, or no binary for this platform
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise RuntimeError("MP4 export needs ffmpeg: `pip install imageio-ffmpeg` (bundles it) "
                       "or install ffmpeg on the PATH")


def _has_encoder(ffmpeg: str, name: str) -> bool:
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    return any(line.split()[1:2] == [name] for line in out.stdout.splitlines() if line.strip())


class Mp4Writer:
    """Encodes a stream of JPEG frames to an MP4 file."""

    def __init__(self, path: str | Path, fps: float, quality: int = 18) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.frames = 0
        ffmpeg = find_ffmpeg()
        if _has_encoder(ffmpeg, "libx264"):
            codec = ["-c:v", "libx264", "-preset", "medium", "-crf", str(quality), "-pix_fmt", "yuv420p"]
        else:
            codec = ["-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p"]
        self._proc = subprocess.Popen(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", f"{self.fps:g}", "-i", "-",
                # yuv420p needs even dimensions.
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                *codec, "-movflags", "+faststart", str(self.path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._lock = threading.Lock()
        self._stderr = b""

    def write(self, jpeg: bytes) -> None:
        with self._lock:
            if self._proc.stdin is None or self._proc.poll() is not None:
                raise RuntimeError(f"ffmpeg stopped: {self._read_stderr()}")
            self._proc.stdin.write(jpeg)
            self.frames += 1

    def _read_stderr(self) -> str:
        if self._proc.stderr is not None and not self._stderr:
            self._stderr = self._proc.stderr.read() or b""
        return self._stderr.decode(errors="replace").strip()

    def close(self) -> Path:
        with self._lock:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            code = self._proc.wait()
            if code != 0:
                raise RuntimeError(f"ffmpeg failed ({code}): {self._read_stderr()}")
        return self.path

    def abort(self) -> None:
        with self._lock:
            if self._proc.poll() is None:
                self._proc.kill()
            self._proc.wait()
        self.path.unlink(missing_ok=True)
