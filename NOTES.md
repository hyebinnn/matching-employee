# 실행 방법 메모

## 준비

- Python 3.12
- [uv](https://docs.astral.sh/uv/) 권장 (macOS: `brew install uv`)

```bash
uv sync                  # .venv 생성 + 의존성 설치 (Python 3.12가 없으면 uv가 받아옴)
cp .env.example .env     # .env에 OPENAI_API_KEY 입력 (커밋 금지)
```

## 서버 실행

```bash
uv run uvicorn app.api.main:app --reload
```

- UI: http://localhost:8000/
- 공고 목록: `GET /jobs`
- 매칭 결과: `GET /jobs/{job_id}/matches` (예: `/jobs/fixture_accountant/matches`)
- API 문서: http://localhost:8000/docs

## 테스트

```bash
uv run pytest
```

테스트는 FakeEmbedder를 사용하므로 OpenAI API 키 없이 실행된다.

## uv 없이 (pip)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt pytest httpx
uvicorn app.api.main:app --reload
pytest
```

의존성을 바꾼 뒤 requirements.txt 갱신:

```bash
uv export --format requirements-txt --no-hashes --no-dev -o requirements.txt
```

## 공고 이미지 파싱 (선택 — 파싱 결과 JSON은 커밋되어 있어 매칭 실행에는 필요 없음)

```bash
brew install tesseract
mkdir -p data/tessdata
curl -L -o data/tessdata/kor.traineddata https://github.com/tesseract-ocr/tessdata_fast/raw/main/kor.traineddata
curl -L -o data/tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata_fast/raw/main/eng.traineddata
```

1. `data/raw/{job_id}/meta.json` (`title`, `company`, `source_url`) 과 `data/raw/{job_id}/images/*` 준비
2. `config/parsing.yaml`의 `vision.model` 설정
3. 실행

```bash
uv run python -m app.parsing.cli {job_id} --dry-run            # 결과만 확인
uv run python -m app.parsing.cli {job_id} --include r03        # 미검증 조건을 확인 후 포함해 저장
```

- 결과: `data/jobs/{job_id}.json`, 검증 리포트 `data/parsed/{job_id}.report.json`, OCR 원문 `data/parsed/{job_id}.ocr.txt`
- 비전 LLM 응답은 `data/cache/parse/`에 캐시되어 재실행 시 과금되지 않고 조건 id도 유지됨

## 데이터 / 설정 위치

- 공고: `data/jobs/{job_id}.json` (파일명 = 공고 id)
- 지원자: `data/resumes/{job_id}/{candidate_id}.json`
- 점수 정책(임계값, 가중치, cap): `config/scoring.yaml` — 수정 후 서버 재시작
- 파싱 설정(비전 모델, 이미지 분할, OCR, 검증 기준): `config/parsing.yaml`
- 임베딩 캐시: `data/cache/embeddings/` — 같은 텍스트는 API를 다시 호출하지 않음. 지우면 다음 실행 때 다시 임베딩
- 지원자 유형 라벨(검증용): `data/expected/{job_id}.json`
