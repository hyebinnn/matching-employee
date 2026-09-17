"""OCR. 조건 추출이 아니라 비전 LLM 결과를 검증하는 원문 텍스트를 얻는 용도다."""

from pathlib import Path
from typing import Protocol

import pytesseract
from PIL import Image, ImageOps

from app.parsing.config import OcrConfig

DEFAULT_TESSDATA_DIR = Path(__file__).resolve().parents[2] / "data" / "tessdata"


class OcrEngine(Protocol):
    def read(self, image: Image.Image) -> str: ...


class TesseractOcr:
    def __init__(self, config: OcrConfig, tessdata_dir: Path = DEFAULT_TESSDATA_DIR) -> None:
        self._config = config
        self._tessdata_dir = tessdata_dir

    def read(self, image: Image.Image) -> str:
        return pytesseract.image_to_string(
            self._preprocess(image),
            lang=self._config.lang,
            config=f'--tessdata-dir "{self._tessdata_dir}" --psm {self._config.psm}',
        )

    def _preprocess(self, image: Image.Image) -> Image.Image:
        gray = ImageOps.grayscale(image)
        if self._config.upscale == 1:
            return gray
        size = (round(gray.width * self._config.upscale), round(gray.height * self._config.upscale))
        return gray.resize(size, Image.Resampling.LANCZOS)
