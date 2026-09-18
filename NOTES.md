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
- 매칭 결과: `GET /jobs/{job_id}/matches` (예: `/jobs/designer/matches`)
- API 문서: http://localhost:8000/docs

커밋된 공고 id는 `designer`, `japan-ecommerce`, `barog-developer` 세 개다.

## 테스트

```bash
uv run pytest
```

테스트는 FakeEmbedder와 OpenAI 클라이언트 대역을 쓰므로 API 키 없이 실행된다.

## 공고 이미지 파싱 (선택 — 파싱 결과 JSON은 커밋되어 있어 매칭 실행에는 필요 없음)

```bash
brew install tesseract
mkdir -p data/tessdata
curl -L -o data/tessdata/kor.traineddata https://github.com/tesseract-ocr/tessdata_fast/raw/main/kor.traineddata
curl -L -o data/tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata_fast/raw/main/eng.traineddata
```

공고 원본 이미지는 타사 저작물이라 커밋하지 않았다(`data/raw/`는 gitignore). 다시 파싱하려면 직접 준비한다.

1. `data/raw/{job_id}/meta.json` (`title`, `company`, `source_url`) 과 `data/raw/{job_id}/images/*` 준비
2. `config/parsing.yaml`의 `vision.model` 설정
3. 실행

```bash
uv run python -m app.parsing.cli {job_id} --dry-run            # 결과만 확인
uv run python -m app.parsing.cli {job_id} --include r03        # 미검증 조건을 확인 후 포함해 저장
```

- 결과: `data/jobs/{job_id}.json`, 검증 리포트 `data/parsed/{job_id}.report.json`, OCR 원문 `data/parsed/{job_id}.ocr.txt`
- 비전 LLM 응답은 `data/cache/parse/`에 캐시되어 재실행 시 과금되지 않고 조건 id도 유지됨

## 품질 채점 / 분석 도구 (임계값 조정 근거)

```bash
uv run python -m tools.evaluate                 # 세 공고 전체 채점
uv run python -m tools.evaluate --holdout       # 검증용 공고(barog-developer)만
uv run python -m tools.evaluate --no-llm        # LLM 재판정 없이 임베딩 판정만
uv run python -m tools.evaluate --sweep         # 임계값 조합 비교 + 안정 구간
uv run python -m tools.export_vectors           # 벡터/히트맵 내보내기
```

- `data/analysis/heatmap.html` — 조건 × 지원자 유사도 표 (칸에 마우스를 올리면 매칭된 이력서 문장)
- `data/analysis/vectors.tsv`, `metadata.tsv` — https://projector.tensorflow.org 에 올려 벡터 공간 확인
- `--sweep`과 `export_vectors`는 캐시된 임베딩만 쓰므로 API를 호출하지 않는다
  (`export_vectors`는 캐시에 없는 텍스트를 만나면 실패한다 — 매칭을 한 번 실행한 뒤 쓴다)

## 데이터 / 설정 위치

- 공고 원본(메타데이터 + 이미지): `data/raw/{job_id}/`
- 공고: `data/jobs/{job_id}.json` (파일명 = 공고 id)
- 지원자: `data/resumes/{job_id}/{candidate_id}.json`
- 직군별 용어 사전: `data/lexicon/*.json` — 파일을 추가하면 바로 반영되고, 없어도 동작은 같음
- 점수 정책(임계값, 가중치, cap, LLM 재판정): `config/scoring.yaml` — 수정 후 서버 재시작
- 파싱 설정(비전 모델, 이미지 분할, OCR, 검증 기준): `config/parsing.yaml`
- 임베딩 캐시: `data/cache/embeddings/` — 같은 텍스트는 API를 다시 호출하지 않음. 지우면 다음 실행 때 다시 임베딩
- LLM 재판정 캐시: `data/cache/judgements/` — (모델, 조건, 이력서) 조합이 같으면 다시 묻지 않음
- 비전 LLM 응답 캐시: `data/cache/parse/` — (모델, 프롬프트, 이미지) 조합이 같으면 다시 호출하지 않음
- 지원자 유형 라벨(검증용): `data/expected/{job_id}.json`
