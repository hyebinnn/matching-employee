"""긴 이미지 조각내기: 빠지는 구간 없이, 경계에 걸린 줄이 한 조각에는 온전히 들어가야 한다."""

import pytest
from PIL import Image

from app.parsing.tiler import split_tall_image, tile_ranges


def test_short_image_is_single_tile():
    assert tile_ranges(height=900, max_height=1600, overlap=120) == [(0, 900)]


def test_tall_image_tiles_cover_whole_height_with_overlap():
    ranges = tile_ranges(height=5000, max_height=1600, overlap=120)

    assert ranges[0][0] == 0
    assert ranges[-1][1] == 5000
    assert all(bottom - top <= 1600 for top, bottom in ranges)
    # 인접 조각은 정확히 overlap만큼 겹친다 → 높이 overlap 이하인 줄은 반드시 한 조각에 온전히 들어간다.
    assert all(prev_bottom - next_top == 120 for (_, prev_bottom), (next_top, _) in zip(ranges, ranges[1:]))


def test_overlap_must_be_smaller_than_tile_height():
    # 같으면 다음 조각 시작점이 앞으로 가지 않아 무한 루프가 된다.
    with pytest.raises(ValueError):
        tile_ranges(height=5000, max_height=100, overlap=100)


def test_split_keeps_width_and_tile_heights():
    tiles = split_tall_image(Image.new("RGB", (800, 3500)), max_height=1600, overlap=120)

    assert [t.size for t in tiles] == [(800, 1600), (800, 1600), (800, 540)]
