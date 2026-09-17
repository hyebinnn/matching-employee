"""세로로 긴 공고 이미지를 겹치는 조각으로 자른다.

비전 모델은 큰 이미지를 축소해서 보므로, 긴 이미지를 통째로 넣으면 글자가 뭉개진다.
"""

from PIL import Image


def tile_ranges(height: int, max_height: int, overlap: int) -> list[tuple[int, int]]:
    """(top, bottom) 구간 목록. 인접 조각은 overlap만큼 겹치고 마지막 조각은 이미지 끝에 닿는다."""
    if overlap >= max_height:
        raise ValueError("overlap은 max_height보다 작아야 함")
    ranges = []
    top = 0
    while True:
        bottom = min(top + max_height, height)
        ranges.append((top, bottom))
        if bottom >= height:
            return ranges
        top = bottom - overlap


def split_tall_image(image: Image.Image, max_height: int, overlap: int) -> list[Image.Image]:
    return [image.crop((0, top, image.width, bottom)) for top, bottom in tile_ranges(image.height, max_height, overlap)]
