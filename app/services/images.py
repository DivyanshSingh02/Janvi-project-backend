"""The upload image pipeline. It has NO database access and does not know about HTTP or cases.

For every upload it: reads at most max_bytes + 1 bytes, detects the real format with Pillow (the client's
Content-Type and file name are never looked at), checks the pixel count BEFORE decoding, applies the EXIF
orientation, flattens transparency onto white, and re-encodes as JPEG. Re-encoding drops all metadata
(GPS position, camera model, ...). The result describes the FINAL bytes.
"""
import hashlib
import io
from dataclasses import dataclass
from typing import BinaryIO

from PIL import Image, ImageOps, UnidentifiedImageError

# MPO is a JPEG container that some phones write; only its first (main) frame is used.
ALLOWED_FORMATS = ("JPEG", "PNG", "WEBP")
_ACCEPTED_FORMAT_NAMES = frozenset({"JPEG", "MPO", "PNG", "WEBP"})
_ANIMATION_CAPABLE = frozenset({"PNG", "WEBP"})
JPEG_QUALITY = 90
_READ_CHUNK = 64 * 1024
_WHITE = (255, 255, 255)
_SIXTEEN_BIT_MODES = frozenset({"I;16", "I;16L", "I;16B", "I;16N", "I"})


class ImageError(Exception):
    """Base class. Messages are fixed strings: they never contain file names or file content."""


class InvalidImageError(ImageError):
    """Not an image, or a corrupt / truncated one."""


class UnsupportedImageFormatError(ImageError):
    """A real image, but not JPEG, PNG or WebP (or an animated one)."""


class ImageTooLargeError(ImageError):
    """Too many bytes or too many pixels."""


class ImageTooSmallError(ImageError):
    """A side is shorter than the minimum."""


@dataclass(frozen=True)
class ImageLimits:
    max_bytes: int
    max_pixels: int
    min_side: int


@dataclass(frozen=True)
class ProcessedImage:
    data: bytes          # the final JPEG
    width: int
    height: int
    sha256: str          # lowercase hex of `data`

    @property
    def size_bytes(self) -> int:
        return len(self.data)


def read_limited(stream: BinaryIO, max_bytes: int) -> bytes:
    """Read the whole stream, but never more than max_bytes + 1 bytes: that one extra byte is enough
    to know the upload is too big, and the rest is never read."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(min(_READ_CHUNK, max_bytes + 1 - total))
        if not chunk:
            break
        total += len(chunk)
        chunks.append(chunk)
        if total > max_bytes:
            raise ImageTooLargeError("The upload is too large")
    return b"".join(chunks)


def process_upload(stream: BinaryIO, limits: ImageLimits) -> ProcessedImage:
    return process_image_bytes(read_limited(stream, limits.max_bytes), limits)


def _open(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data), formats=list(ALLOWED_FORMATS))
    except Image.DecompressionBombError:
        raise ImageTooLargeError("The image has too many pixels") from None
    except UnidentifiedImageError:
        pass
    except (OSError, SyntaxError, ValueError, EOFError, ArithmeticError):
        raise InvalidImageError("The file is not a valid image") from None
    # Not one of the allowed formats. Is it some other real image (GIF, BMP, ...) or not an image at all?
    try:
        other = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError:
        raise ImageTooLargeError("The image has too many pixels") from None
    except Exception:
        raise InvalidImageError("The file is not a valid image") from None
    other.close()
    raise UnsupportedImageFormatError("Only JPEG, PNG and WebP images are accepted")


def _to_rgb(image: Image.Image) -> Image.Image:
    mode = image.mode
    if mode == "RGB":
        return image
    if mode in _SIXTEEN_BIT_MODES:
        # 16-bit grey: convert() would clip everything above 255 to white, so scale down first.
        return image.convert("I").point(lambda value: value / 256).convert("L").convert("RGB")
    if mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        flattened = Image.new("RGB", rgba.size, _WHITE)
        flattened.paste(rgba, mask=rgba.getchannel("A"))
        return flattened
    return image.convert("RGB")


def process_image_bytes(data: bytes, limits: ImageLimits) -> ProcessedImage:
    if len(data) > limits.max_bytes:
        raise ImageTooLargeError("The upload is too large")
    image = _open(data)
    try:
        if image.format in _ANIMATION_CAPABLE and getattr(image, "is_animated", False):
            raise UnsupportedImageFormatError("Animated images are not accepted")
        width, height = image.size                     # read from the header: nothing is decoded yet
        if width * height > limits.max_pixels:
            raise ImageTooLargeError("The image has too many pixels")
        if min(width, height) < limits.min_side:
            raise ImageTooSmallError("The image is too small")
        try:
            image.load()                               # the real decode; corrupt data fails here
        except Image.DecompressionBombError:
            raise ImageTooLargeError("The image has too many pixels") from None
        except MemoryError:
            raise ImageTooLargeError("The image has too many pixels") from None
        except (OSError, SyntaxError, ValueError, EOFError, ArithmeticError):
            raise InvalidImageError("The file is not a valid image") from None

        oriented = ImageOps.exif_transpose(image)      # a rotated copy; the orientation tag is dropped
        rgb = _to_rgb(oriented)
        output = io.BytesIO()
        # Nothing is passed on from the source (no exif=, no icc_profile=), so every metadata block
        # of the original is gone from the saved file.
        rgb.save(output, format="JPEG", quality=JPEG_QUALITY)
        final = output.getvalue()
        return ProcessedImage(
            data=final, width=rgb.width, height=rgb.height, sha256=hashlib.sha256(final).hexdigest()
        )
    finally:
        image.close()
