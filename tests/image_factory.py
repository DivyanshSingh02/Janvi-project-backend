"""Builds small, entirely fictional images for the tests (there are no real photos in this project).
Plain functions with no dependency on the app, so conftest can expose them as fixtures."""
import io
import random
import struct
import zlib

from PIL import Image, ImageDraw


def _picture(size: tuple[int, int], seed: int, mode: str = "RGB") -> Image.Image:
    """A deterministic picture of coloured rectangles. A different seed gives different pixels,
    which gives a different SHA-256 after re-encoding."""
    rng = random.Random(seed)
    image = Image.new("RGB", size, (rng.randrange(256), rng.randrange(256), rng.randrange(256)))
    draw = ImageDraw.Draw(image)
    for _ in range(6):
        x1, y1 = rng.randrange(size[0]), rng.randrange(size[1])
        x2, y2 = rng.randrange(size[0]), rng.randrange(size[1])
        draw.rectangle(
            [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
            fill=(rng.randrange(256), rng.randrange(256), rng.randrange(256)),
        )
    if mode == "RGB":
        return image
    if mode == "RGBA":
        rgba = image.convert("RGBA")
        rgba.paste((0, 0, 0, 0), [0, 0, size[0] // 2, size[1] // 2])      # top-left quarter fully transparent
        return rgba
    if mode == "LA":
        alpha = Image.new("L", size, 255)
        alpha.paste(0, [0, 0, size[0] // 2, size[1] // 2])                # top-left quarter fully transparent
        return Image.merge("LA", (image.convert("L"), alpha))
    if mode == "P":
        return image.convert("P")
    return image.convert(mode)


def make_image_bytes(
    fmt: str = "JPEG",
    size: tuple[int, int] = (200, 200),
    seed: int = 0,
    mode: str = "RGB",
    orientation: int | None = None,
    with_metadata: bool = False,
    frames: int = 1,
) -> bytes:
    """fmt: JPEG, PNG, WEBP, GIF, BMP or MPO. orientation: an EXIF orientation value (1-8).
    with_metadata: adds a fictional camera make and a fictional GPS position to the EXIF block.
    frames > 1: an animated image (PNG, WEBP or GIF)."""
    image = _picture(size, seed, mode)
    save_args: dict = {}
    if orientation is not None or with_metadata:
        exif = Image.Exif()
        if orientation is not None:
            exif[0x0112] = orientation
        if with_metadata:
            exif[0x010F] = "FakeCam Industries"
            exif[0x0110] = "FakeCam 9000"
            exif[0x8825] = {1: "N", 2: (12.0, 34.0, 56.0), 3: "E", 4: (78.0, 9.0, 10.0)}
        save_args["exif"] = exif
    if fmt == "JPEG":
        save_args["quality"] = 85
    if frames > 1:
        others = [_picture(size, seed + index, mode) for index in range(1, frames)]
        save_args.update(save_all=True, append_images=others)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **save_args)
    return buffer.getvalue()


def make_png_with_declared_size(width: int, height: int) -> bytes:
    """A tiny PNG whose header CLAIMS a large size (like a decompression bomb). Opening it reads only
    the header, so the test never needs to allocate the pixels."""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)       # 8-bit greyscale
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"\x00" * 32)) + chunk(b"IEND", b"")


def make_noisy_png(size: tuple[int, int] = (300, 300), seed: int = 0) -> bytes:
    """Random pixels do not compress, so this PNG is big: handy for 'too many bytes' tests."""
    rng = random.Random(seed)
    image = Image.frombytes("RGB", size, bytes(rng.randrange(256) for _ in range(size[0] * size[1] * 3)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
