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

## 데이터 / 설정 위치

- 공고: `data/jobs/{job_id}.json` (파일명 = 공고 id)
- 지원자: `data/resumes/{job_id}/{candidate_id}.json`
- 점수 정책(임계값, 가중치, cap): `config/scoring.yaml` — 수정 후 서버 재시작
- 임베딩 캐시: `data/cache/embeddings/` — 같은 텍스트는 API를 다시 호출하지 않음. 지우면 다음 실행 때 다시 임베딩
- `fixture_` 접두사 데이터는 파이프라인 확인용 임시 데이터 (제출용 목업 이력서 아님)
