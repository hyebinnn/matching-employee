"""매칭 품질 평가: 목업 이력서에 붙여둔 의도 유형(완벽/부분/미스매칭) 순서가 지켜지는지 센다.

    uv run python -m tools.evaluate              # 현재 설정 채점
    uv run python -m tools.evaluate --no-llm     # 임베딩 판정만으로 채점
    uv run python -m tools.evaluate --sweep      # 임계값 조합 비교 + 안정 구간 (임베딩 판정만, API 호출 없음)
    uv run python -m tools.evaluate --holdout    # 튜닝에 쓰지 않은 검증용 공고만 채점

--sweep은 최적값 하나가 아니라 **안정 구간**(순위 역전이 최소인 임계값이 몇 개나 연속으로 이어지는지)도
출력한다. 한 점에서만 결과가 좋다면 그 값은 데이터에 우연히 맞은 것이고, 넓은 구간에서 같은 결과가
나온다면 값이 다소 달라져도 견딘다는 뜻이다.

채점 기준은 "순위 역전": 더 잘 맞아야 할 유형이 덜 맞는 유형보다 아래에 있는 쌍의 수.
0이면 의도한 순서가 모두 지켜진 것이다. 점수의 절대값이 아니라 순서만 본다.

--sweep은 캐시된 임베딩만 사용하므로 몇 초면 끝나고 과금되지 않는다. (임계값은 이미 계산된
유사도를 어디서 자를지 정하는 값이라, 조합을 바꿔도 임베딩을 다시 계산할 필요가 없다)
"""

import argparse
import json
from itertools import product
from pathlib import Path

from dotenv import load_dotenv

from app.cache import EmbeddingCache, JsonFileCache
from app.domain.models import MatchResult
from app.matching.config import ScoringConfig, load_config
from app.matching.embedder import CachedEmbedder, OpenAIEmbedder
from app.matching.judge import EmbeddingJudge, RequirementJudge, YearsJudge
from app.matching.lexicon import Lexicon
from app.matching.llm_judge import LlmJudge
from app.matching.service import MatchingService
from app.repository import JsonRepository

# 잘 맞는 순서. 값이 작을수록 위에 있어야 한다.
TYPE_ORDER = {"perfect": 0, "partial": 1, "mismatch": 2}
MET_CANDIDATES = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
PARTIAL_CANDIDATES = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]


def build_service(config: ScoringConfig, use_llm: bool) -> MatchingService:
    embedder = CachedEmbedder(OpenAIEmbedder(config.embedding.model), EmbeddingCache())
    text_judge: RequirementJudge = EmbeddingJudge(embedder, config.thresholds, Lexicon.load())
    if use_llm and config.llm_judge.enabled:
        text_judge = LlmJudge(text_judge, config.llm_judge, JsonFileCache())
    return MatchingService(text_judge, YearsJudge(config.years), config)


def load_labels(job_id: str) -> dict[str, str]:
    data = _expected(job_id)["candidates"]
    return {cid: info["type"] for cid, info in data.items()}


def _expected(job_id: str) -> dict:
    return json.loads(Path(f"data/expected/{job_id}.json").read_text(encoding="utf-8"))


def is_holdout(job_id: str) -> bool:
    """임계값 조정에 쓰지 않은 검증용 공고인지."""
    return _expected(job_id).get("purpose") == "holdout"


def inversions(ranked: list[MatchResult], labels: dict[str, str]) -> int:
    """순위 역전 쌍의 수."""
    order = [TYPE_ORDER[labels[m.candidate_id]] for m in ranked]
    return sum(1 for i, above in enumerate(order) for below in order[i + 1 :] if below < above)


def margin(ranked: list[MatchResult], labels: dict[str, str]) -> float:
    """완벽 매칭 최저점 − 미스매칭 최고점. 클수록 유형이 잘 갈린다."""
    perfect = [m.score for m in ranked if labels[m.candidate_id] == "perfect"]
    mismatch = [m.score for m in ranked if labels[m.candidate_id] == "mismatch"]
    if not perfect or not mismatch:
        return float("nan")
    return min(perfect) - max(mismatch)


def evaluate(service: MatchingService, repo: JsonRepository, only: str | None = None) -> list[tuple]:
    """only: "holdout"이면 검증용 공고만, "tuning"이면 튜닝에 쓴 공고만 채점한다."""
    results = []
    for job in repo.list_jobs():
        if only == "holdout" and not is_holdout(job.id):
            continue
        if only == "tuning" and is_holdout(job.id):
            continue
        labels = load_labels(job.id)
        ranked = service.rank(job, repo.list_candidates(job.id))
        results.append((job, ranked, labels, inversions(ranked, labels), margin(ranked, labels)))
    return results


def report(results: list[tuple]) -> None:
    for job, ranked, labels, bad, gap in results:
        tag = " [검증용 — 임계값 조정에 사용하지 않음]" if is_holdout(job.id) else ""
        print(f"\n== {job.title} ({job.id}) — 순위 역전 {bad}회, 완벽-미스매칭 간격 {gap:.1f}점{tag}")
        for rank, m in enumerate(ranked, start=1):
            cap = f"  (상한 {m.applied_cap:.0f} 적용)" if m.applied_cap is not None else ""
            print(f"  {rank}. {m.candidate_name:6} {m.score:5.1f}  [{labels[m.candidate_id]}]{cap}")
    total = sum(r[3] for r in results)
    print(f"\n전체 순위 역전: {total}회")


def sweep(config: ScoringConfig, repo: JsonRepository) -> None:
    """임계값 조합별 순위 역전 횟수. 임베딩 판정만 사용한다(임계값이 직접 영향을 주는 경로)."""
    rows, grid = [], {}
    for met, partial in product(MET_CANDIDATES, PARTIAL_CANDIDATES):
        if partial > met:
            continue
        tuned = ScoringConfig.model_validate({**config.model_dump(), "thresholds": {"met": met, "partial": partial}})
        results = evaluate(build_service(tuned, use_llm=False), repo, only="tuning")
        bad = sum(r[3] for r in results)
        rows.append((bad, -min(r[4] for r in results), met, partial))
        grid[(met, partial)] = bad

    _print_plateau(grid)
    rows.sort()
    current = (config.thresholds.met, config.thresholds.partial)
    print("임계값 조합 비교 (임베딩 판정만, 순위 역전이 적을수록 좋음)\n")
    print("  met  partial | 역전 | 완벽-미스매칭 간격")
    for bad, neg_gap, met, partial in rows:
        mark = "  ← 현재 설정" if (met, partial) == current else ""
        print(f"  {met:.2f}  {partial:.2f}   |  {bad:2}   | {-neg_gap:6.1f}{mark}")
    print(f"\n조합 {len(rows)}개 비교. 캐시된 임베딩만 사용하므로 API 호출은 없다.")


def _print_plateau(grid: dict[tuple[float, float], int]) -> None:
    """임계값 격자를 표로 그려 안정 구간(최소 역전이 이어지는 영역)을 보여준다."""
    best = min(grid.values())
    print(f"\n안정 구간 (순위 역전 최소 = {best}회, ■ = 최소인 조합)\n")
    header = "  met＼partial |" + "".join(f" {p:>5.2f}" for p in PARTIAL_CANDIDATES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for met in MET_CANDIDATES:
        cells = []
        for partial in PARTIAL_CANDIDATES:
            bad = grid.get((met, partial))
            cells.append("     ·" if bad is None else (f"    ■{bad}" if bad == best else f"     {bad}"))
        print(f"  {met:>11.2f} |" + "".join(cells))
    tolerant = sorted(k for k, v in grid.items() if v == best)
    print(f"\n  최소 역전 조합 {len(tolerant)}개 / 전체 {len(grid)}개")
    print(f"  met {min(m for m, _ in tolerant):.2f}~{max(m for m, _ in tolerant):.2f}, "
          f"partial {min(p for _, p in tolerant):.2f}~{max(p for _, p in tolerant):.2f} 범위에서 같은 결과")


def main() -> None:
    parser = argparse.ArgumentParser(description="목업 이력서 라벨로 매칭 품질을 채점한다")
    parser.add_argument("--sweep", action="store_true", help="임계값 조합 비교")
    parser.add_argument("--no-llm", action="store_true", help="LLM 재판정 없이 임베딩 판정만으로 채점")
    parser.add_argument("--holdout", action="store_true", help="임계값 조정에 쓰지 않은 검증용 공고만 채점")
    args = parser.parse_args()

    load_dotenv()
    config = load_config()
    repo = JsonRepository()

    if args.sweep:
        sweep(config, repo)
        return
    report(evaluate(build_service(config, use_llm=not args.no_llm), repo, only="holdout" if args.holdout else None))


if __name__ == "__main__":
    main()
