"""The one door a picture takes into the engineer's conversation (Phase 64).

A picture reaches the brain as a PATH - one the user attached in the composer (moved into the
edit's ``chats/attachments/`` when the message was sent) or one the app made for the agent's
``view.snapshot`` (the app's picture cache). This module reads it, checks it and turns it into
the image block the Messages API takes. Nothing else in the brain opens a picture.

What it holds a picture to, every one a sentence when it refuses:

- **Where it is.** Only those two folders. The chat request and a command's reply both come
  from the app, so this is defence in depth: a path that came from anywhere else - a project
  file, a model's argument - is never read and sent to Anthropic.
- **What it is.** PNG or JPEG by its first bytes, never by its name. The app turns every other
  format (HEIC, WebP, TIFF, GIF) into one of the two before it is attached.
- **How big it is.** Long edge at most 1568 px and at most 1.15 megapixels - the size Anthropic
  sends to the model without resizing it - and 3.75 MB of file (5 MB once in base64). The app
  scales every picture under this; a picture over it was not made by the app.
- **How many.** Five to a message.

A picture costs about `width x height / 750` input tokens; at the caps that is 1,533, so the
history's estimate counts each one as `TOKENS_PER_PICTURE`.
"""
from __future__ import annotations

import base64
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .llm_backend import NoRetry

MAX_PICTURES = 5                 # in one message
MAX_LONG_EDGE_PX = 1568
MAX_PIXELS = 1_150_000
MAX_BYTES = 3_750_000            # 5 MB once base64-encoded: the API's limit for one image
TOKENS_PER_PICTURE = 1_600       # the ceiling of w x h / 750 at the caps, rounded up

#: Where a picture the user attached lives once its message was sent: beside the transcripts
#: (`chats/<id>.json`, ChatSessions.h), under the edit's brain folder.
ATTACHMENTS = ("chats", "attachments")
#: The app's picture cache (`agentpictures::cacheDir`, AgentPictures.h): where a picture waits
#: in the composer, and where the agent's snapshots are written. The same override as the app.
CACHE_ENV = "ZAMMOTH_PICTURES_DIR"


#: Where a command's reply carries the pictures the brain read for it (`daw_bridge.
#: attach_snapshot`) on its way to the tool result: popped before the reply is JSON.
RESULT_KEY = "_pictures"


class PictureRefused(NoRetry):
    """A picture the brain will not send. Asking again repeats it: the sentence says why."""


@dataclass(frozen=True)
class Picture:
    path: Path
    media_type: str
    width: int
    height: int
    data: bytes

    def block(self) -> dict:
        """The Messages API's image block."""
        return {"type": "image",
                "source": {"type": "base64", "media_type": self.media_type,
                           "data": base64.b64encode(self.data).decode("ascii")}}


def cache_dir() -> Path:
    """The app's picture cache, as the app computes it."""
    over = os.environ.get(CACHE_ENV, "").strip()
    if over and os.path.isabs(over):
        return Path(over)
    return Path.home() / "Library" / "Caches" / "Zammoth" / "Pictures"


def attachments_dir(project_dir: Path) -> Path:
    return Path(project_dir).joinpath(*ATTACHMENTS)


def _inside(path: Path, folder: Path) -> bool:
    try:
        path.relative_to(folder.resolve())
        return True
    except (ValueError, OSError):
        return False


def _png_size(data: bytes) -> Optional[tuple[int, int]]:
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    return None


#: JPEG markers that are a frame header (SOF0-SOF15 but the three that are not frames).
_SOF = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


def _jpeg_size(data: bytes) -> Optional[tuple[int, int]]:
    if len(data) < 4 or data[:3] != b"\xff\xd8\xff":
        return None
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:              # fill byte
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:   # no length
            i += 2
            continue
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        if length < 2:
            return None
        if marker in _SOF:
            if i + 9 > len(data):
                return None
            height, width = struct.unpack(">HH", data[i + 5:i + 9])
            return int(width), int(height)
        i += 2 + length
    return None


def read(path_text: str, project_dir: Path) -> Picture:
    """One picture, checked. Raises `PictureRefused` with the sentence the user is shown."""
    raw = str(path_text or "").strip()
    name = os.path.basename(raw) or "the picture"
    if not raw or not os.path.isabs(raw):
        raise PictureRefused(f"{name}: a picture is sent by its full path, and this one has none.")
    try:
        path = Path(raw).resolve(strict=True)
    except (OSError, RuntimeError):
        raise PictureRefused(f"{name} is not there any more - attach it again.") from None
    if not (_inside(path, attachments_dir(project_dir)) or _inside(path, cache_dir())):
        raise PictureRefused(f"{name} is not one the composer prepared, so it was not sent.")
    try:
        size = path.stat().st_size
    except OSError:
        raise PictureRefused(f"{name} could not be read.") from None
    if size > MAX_BYTES:
        raise PictureRefused(f"{name} is {size / 1e6:.1f} MB; a picture may be at most "
                             f"{MAX_BYTES / 1e6:.2f} MB.")
    try:
        data = path.read_bytes()
    except OSError:
        raise PictureRefused(f"{name} could not be read.") from None
    if (dims := _png_size(data)) is not None:
        media_type = "image/png"
    elif (dims := _jpeg_size(data)) is not None:
        media_type = "image/jpeg"
    else:
        raise PictureRefused(f"{name} is not a PNG or JPEG picture.")
    width, height = dims
    if width <= 0 or height <= 0:
        raise PictureRefused(f"{name} has no size.")
    if max(width, height) > MAX_LONG_EDGE_PX or width * height > MAX_PIXELS:
        raise PictureRefused(f"{name} is {width} x {height}; a picture may be at most "
                             f"{MAX_LONG_EDGE_PX} px on its long edge and "
                             f"{MAX_PIXELS / 1e6:.2f} megapixels.")
    return Picture(path=path, media_type=media_type, width=width, height=height, data=data)


def read_all(paths: Iterable[str], project_dir: Path) -> list[Picture]:
    """Every picture of one message, in order: all of them or a refusal."""
    listed = [p for p in (paths or []) if str(p or "").strip()]
    if len(listed) > MAX_PICTURES:
        raise PictureRefused(f"{len(listed)} pictures were attached; a message takes at most "
                             f"{MAX_PICTURES}.")
    return [read(p, project_dir) for p in listed]


def transport_refusal(backend: str) -> Optional[str]:
    """Why a turn with pictures cannot go out on this transport, or None when it can. Only the
    direct API (the user's own key) carries them: the development transport hands the SDK one
    text prompt, and the dormant metered proxy refuses image blocks (backend/app/proxy.py)."""
    if backend == "api":
        return None
    if backend == "subscription":
        return ("Pictures need the Anthropic API; the development transport sends text only. "
                "Send the message without them.")
    return "Pictures are not carried on this connection yet. Send the message without them."


def without_picture_data(value):
    """`value` with every image block's data left out, and how many image blocks there were -
    for an estimate of a history's size that does not read base64 as text."""
    count = 0

    def walk(node):
        nonlocal count
        if isinstance(node, dict):
            if node.get("type") == "image":
                count += 1
                return {"type": "image"}
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(value), count
