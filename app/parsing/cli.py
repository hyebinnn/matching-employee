"""공고 이미지 → data/jobs/{job_id}.json

    uv run python -m app.parsing.cli {job_id} [--dry-run] [--include r03 --include r05]

1. data/raw/{job_id}의 이미지를 조각내 비전 LLM으로 조건을 구조화한다.
2. 원본 이미지를 OCR해 조건 문장이 원문에 있는지 검증한다.
3. 검증되지 않은 조건은 목록으로 보여주고 기본 제외한다. 사람이 확인 후 --include로 포함시킨다.
   (LLM 응답이 캐시되므로 다시 실행해도 조건 id가 바뀌지 않는다.)
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from app.domain.models import JobPosting, Requirement
from app.ingest.source import LocalImageSource
from app.parsing.config import load_parsing_config
from app.parsing.ocr import TesseractOcr
from app.parsing.tiler import split_tall_image
from app.parsing.verify import verify_text
from app.parsing.vision import VisionParser

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="공고 이미지를 파싱해 JobPosting JSON으로 저장")
    parser.add_argument("job_id")
    parser.add_argument("--include", action="append", default=[], metavar="REQ_ID", help="OCR로 검증되지 않았지만 포함할 조건 id")
    parser.add_argument("--dry-run", action="store_true", help="결과만 출력하고 저장하지 않음")
    args = parser.parse_args(argv)

    load_dotenv()
    config = load_parsing_config()
    if not config.vision.model:
        print("config/parsing.yaml의 vision.model을 먼저 설정하세요.", file=sys.stderr)
        return 1

    raw = LocalImageSource().load(args.job_id)
    images = [Image.open(path) for path in raw.image_paths]
    tiles = [tile for image in images for tile in split_tall_image(image, config.tiling.max_height, config.tiling.overlap)]

    extracted = VisionParser(config.vision.model).parse(tiles)
    requirements = [
        Requirement(id=f"r{i:02d}", category=e.category, text=e.text, type=e.type, min_years=e.min_years)
        for i, e in enumerate(extracted, start=1)
    ]

    ocr = TesseractOcr(config.ocr)
    ocr_text = "\n".join(ocr.read(image) for image in images)
    verifications = {
        r.id: verify_text(r.id, r.text, ocr_text, config.verify.min_similarity) for r in requirements
    }

    unknown = set(args.include) - {r.id for r in requirements}
    if unknown:
        print(f"없는 조건 id: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 1

    included = [r for r in requirements if verifications[r.id].verified or r.id in args.include]
    _print_summary(requirements, verifications, set(args.include))

    if args.dry_run:
        print("\n--dry-run: 저장하지 않음")
        return 0
    if not included:
        print("\n포함된 조건이 없어 저장하지 않음", file=sys.stderr)
        return 1

    job = JobPosting(
        id=raw.job_id, title=raw.title, company=raw.company, source_url=raw.source_url, requirements=included
    )
    _write(DATA_DIR / "jobs" / f"{raw.job_id}.json", job.model_dump(mode="json"))
    _write(
        DATA_DIR / "parsed" / f"{raw.job_id}.report.json",
        {
            "job_id": raw.job_id,
            "vision_model": config.vision.model,
            "ocr": config.ocr.model_dump(),
            "verify": config.verify.model_dump(),
            "image_count": len(images),
            "tile_count": len(tiles),
            "requirements": [
                {
                    **r.model_dump(mode="json"),
                    "ocr_similarity": round(verifications[r.id].similarity, 3),
                    "verified": verifications[r.id].verified,
                    "matched_ocr_text": verifications[r.id].matched_text,
                    "included": r in included,
                }
                for r in requirements
            ],
        },
    )
    (DATA_DIR / "parsed" / f"{raw.job_id}.ocr.txt").write_text(ocr_text, encoding="utf-8")
    print(f"\n저장: data/jobs/{raw.job_id}.json (조건 {len(included)}/{len(requirements)}개), data/parsed/{raw.job_id}.*")
    return 0


def _print_summary(requirements, verifications, include_ids: set[str]) -> None:
    for r in requirements:
        v = verifications[r.id]
        mark = "검증" if v.verified else ("포함(수동)" if r.id in include_ids else "제외(미검증)")
        years = f" min_years={r.min_years}" if r.min_years is not None else ""
        print(f"[{mark}] {r.id} {r.type.value}/{r.category.value}{years} ocr={v.similarity:.2f} | {r.text}")
        if not v.verified:
            print(f"        OCR 최근접: {v.matched_text!r}")


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
