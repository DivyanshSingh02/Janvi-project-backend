"""The image pipeline on its own (no database, no HTTP)."""
import hashlib
from io import BytesIO

import pytest
from PIL import Image

from app.services import images

LIMITS = images.ImageLimits(max_bytes=5 * 1024 * 1024, max_pixels=25_000_000, min_side=100)


def _process(data: bytes, limits: images.ImageLimits = LIMITS) -> images.ProcessedImage:
    return images.process_image_bytes(data, limits)


def test_jpeg_png_and_webp_all_come_out_as_jpeg_with_a_matching_description(make_image_bytes):
    for fmt in ("JPEG", "PNG", "WEBP"):
        result = _process(make_image_bytes(fmt, size=(240, 160)))
        opened = Image.open(BytesIO(result.data))
        assert opened.format == "JPEG"
        assert (result.width, result.height) == opened.size == (240, 160)
        assert result.sha256 == hashlib.sha256(result.data).hexdigest()
        assert result.size_bytes == len(result.data)


def test_transparent_areas_are_flattened_onto_white(make_image_bytes):
    for mode in ("RGBA", "LA"):
        result = _process(make_image_bytes("PNG", mode=mode))
        transparent = Image.open(BytesIO(result.data)).getpixel((5, 5))           # inside the transparent quarter
        opaque = Image.open(BytesIO(result.data)).getpixel((190, 190))             # outside it
        assert all(channel >= 250 for channel in transparent), mode
        assert not all(channel >= 250 for channel in opaque), mode                  # the rest of the picture is kept


def test_exif_orientation_is_applied(make_image_bytes):
    rotated = _process(make_image_bytes("JPEG", size=(200, 100), orientation=6))
    assert (rotated.width, rotated.height) == (100, 200)
    upside_down = _process(make_image_bytes("JPEG", size=(200, 100), orientation=3))
    assert (upside_down.width, upside_down.height) == (200, 100)


def test_all_metadata_including_gps_is_removed(make_image_bytes):
    source = make_image_bytes("JPEG", with_metadata=True)
    assert Image.open(BytesIO(source)).getexif().get_ifd(0x8825)       # the fictional GPS tags really are in the input
    result = _process(source)
    stored = Image.open(BytesIO(result.data))
    assert len(stored.getexif()) == 0
    assert not stored.getexif().get_ifd(0x8825)
    assert b"FakeCam" not in result.data
    assert b"Exif" not in result.data


def test_a_text_file_and_random_bytes_are_not_images():
    for data in (b"plain text pretending to be a photo\n" * 20, bytes(range(256)) * 40, b""):
        with pytest.raises(images.InvalidImageError):
            _process(data)


def test_a_real_image_in_another_format_is_unsupported(make_image_bytes):
    for fmt in ("GIF", "BMP"):
        with pytest.raises(images.UnsupportedImageFormatError):
            _process(make_image_bytes(fmt))


def test_animated_images_are_rejected(make_image_bytes):
    for fmt in ("PNG", "WEBP", "GIF"):
        with pytest.raises(images.UnsupportedImageFormatError):
            _process(make_image_bytes(fmt, frames=3))


def test_truncated_and_corrupt_images_are_invalid(make_image_bytes):
    jpeg = make_image_bytes("JPEG")
    png = make_image_bytes("PNG")
    for damaged in (jpeg[: len(jpeg) // 2], png[: len(png) // 2], jpeg[:2] + bytes(range(256)) * 20):
        with pytest.raises(images.InvalidImageError):
            _process(damaged)


def test_the_stream_is_never_read_past_the_limit_plus_one_byte():
    class CountingStream(BytesIO):
        bytes_served = 0

        def read(self, size=-1):
            chunk = super().read(size)
            self.bytes_served += len(chunk)
            return chunk

    stream = CountingStream(b"x" * 2_000_000)
    with pytest.raises(images.ImageTooLargeError):
        images.read_limited(stream, 1000)
    assert stream.bytes_served == 1001                                  # the limit plus the one byte that proves it

    exact = CountingStream(b"x" * 1000)
    assert len(images.read_limited(exact, 1000)) == 1000                # exactly at the limit is fine


def test_too_many_pixels_and_decompression_bombs_are_too_large(make_png_with_declared_size, make_image_bytes):
    for width, height in ((6000, 6000), (20000, 20000)):               # 36 million pixels; 400 million pixels
        with pytest.raises(images.ImageTooLargeError):
            _process(make_png_with_declared_size(width, height))
    tiny_limits = images.ImageLimits(max_bytes=LIMITS.max_bytes, max_pixels=10_000, min_side=100)
    with pytest.raises(images.ImageTooLargeError):                      # a real 200x200 image over a 10,000 pixel limit
        _process(make_image_bytes("PNG", size=(200, 200)), tiny_limits)


def test_the_pixel_limit_is_checked_before_decoding(make_png_with_declared_size):
    # The pixel data of this file is garbage, so decoding would say "invalid image". Getting
    # "too large" proves the size check came first.
    with pytest.raises(images.ImageTooLargeError):
        _process(make_png_with_declared_size(6000, 6000))
    with pytest.raises(images.InvalidImageError):
        _process(make_png_with_declared_size(5000, 5000))               # exactly 25 million pixels passes the size check


def test_each_side_must_reach_the_minimum(make_image_bytes):
    for size in ((99, 200), (200, 99), (50, 50)):
        with pytest.raises(images.ImageTooSmallError):
            _process(make_image_bytes("PNG", size=size))
    assert _process(make_image_bytes("PNG", size=(100, 100))).width == 100


def test_sixteen_bit_greyscale_is_scaled_not_clipped_to_white():
    gradient = Image.new("I;16", (120, 120))
    for x in range(120):
        for y in range(120):
            gradient.putpixel((x, y), x * 500)
    buffer = BytesIO()
    gradient.save(buffer, format="PNG")
    result = _process(buffer.getvalue())
    converted = Image.open(BytesIO(result.data))
    assert converted.getpixel((2, 60))[0] < 40
    assert 80 < converted.getpixel((60, 60))[0] < 160
    assert converted.getpixel((119, 60))[0] > 200


def test_a_multi_picture_jpeg_is_treated_as_a_jpeg(make_image_bytes):
    result = _process(make_image_bytes("MPO", frames=2))
    assert Image.open(BytesIO(result.data)).format == "JPEG"


def test_identical_input_gives_an_identical_hash_and_different_pictures_differ(make_image_bytes):
    first = _process(make_image_bytes("JPEG", seed=7))
    again = _process(make_image_bytes("JPEG", seed=7))
    other = _process(make_image_bytes("JPEG", seed=8))
    assert first.sha256 == again.sha256
    assert first.sha256 != other.sha256
