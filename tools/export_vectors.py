"""캐시된 임베딩으로 분석 자료를 만든다. OpenAI를 호출하지 않는다 (캐시에 없으면 실패).

    uv run python -m tools.export_vectors [--out data/analysis]

산출물:
- vectors.tsv / metadata.tsv : https://projector.tensorflow.org 에 올려 벡터 공간을 직접 보는 용도
- heatmap.html               : 조건 × 지원자 유사도 표. 임계값을 어디에 둘지 판단한 근거

연차 조건(min_years)은 임베딩을 쓰지 않으므로 제외한다.
"""

import argparse
import html
import json
from pathlib import Path

from app.cache import EmbeddingCache
from app.matching.chunker import split_sentences
from app.matching.config import ScoringConfig, load_config
from app.matching.judge import cosine
from app.repository import JsonRepository

DEFAULT_OUT = Path("data/analysis")


def main() -> None:
    parser = argparse.ArgumentParser(description="임베딩 캐시 → Projector TSV + 유사도 히트맵")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)

    config = load_config()
    cache = EmbeddingCache()
    repo = JsonRepository()

    def vector(text: str) -> list[float]:
        found = cache.get(config.embedding.model, text)
        if found is None:
            raise SystemExit(f"임베딩 캐시에 없음(먼저 매칭을 한 번 실행하세요): {text[:40]}")
        return found

    rows, vectors, jobs_data = [], [], []
    for job in repo.list_jobs():
        text_requirements = [r for r in job.requirements if r.min_years is None]
        for requirement in text_requirements:
            rows.append(("조건", job.id, "-", requirement.type.value, requirement.text))
            vectors.append(vector(requirement.text))

        labels = json.loads(Path(f"data/expected/{job.id}.json").read_text(encoding="utf-8"))["candidates"]
        matrix = []
        for candidate in repo.list_candidates(job.id):
            chunks = split_sentences(candidate.resume_text)
            chunk_vectors = [vector(c) for c in chunks]
            label = labels.get(candidate.id, {}).get("type", "-")
            for chunk, chunk_vector in zip(chunks, chunk_vectors, strict=True):
                rows.append(("이력서", job.id, candidate.id, label, chunk))
                vectors.append(chunk_vector)

            column = []
            for requirement in job.requirements:
                if requirement.min_years is not None:
                    column.append((None, "연차 조건 (임베딩 미사용)"))
                    continue
                sims = [cosine(vector(requirement.text), cv) for cv in chunk_vectors]
                best = max(range(len(chunks)), key=sims.__getitem__)
                column.append((sims[best], chunks[best]))
            matrix.append((candidate, label, column))
        jobs_data.append((job, matrix))

    _write_tsv(out, rows, vectors)
    (out / "heatmap.html").write_text(_heatmap(jobs_data, config), encoding="utf-8")
    print(f"{out}/vectors.tsv, metadata.tsv, heatmap.html — 벡터 {len(vectors)}개 × {len(vectors[0])}차원")


def _write_tsv(out: Path, rows: list[tuple], vectors: list[list[float]]) -> None:
    with (out / "vectors.tsv").open("w", encoding="utf-8") as f:
        for v in vectors:
            f.write("\t".join(f"{x:.6f}" for x in v) + "\n")
    with (out / "metadata.tsv").open("w", encoding="utf-8") as f:
        f.write("kind\tjob\tcandidate\tlabel\ttext\n")
        for row in rows:
            f.write("\t".join(str(x).replace("\t", " ") for x in row) + "\n")


def _cell_style(similarity: float | None, config: ScoringConfig) -> str:
    if similarity is None:
        return "background:#eee;color:#888"
    if similarity >= config.thresholds.met:
        return "background:#1f7a3f;color:#fff"
    if similarity >= config.thresholds.partial:
        return f"background:rgba(230,180,60,{0.35 + similarity:.2f})"
    if similarity >= config.llm_judge.band.low:
        return "background:#f3e6cc"
    return "background:#f8e3e3"


def _heatmap(jobs_data: list, config: ScoringConfig) -> str:
    t = config.thresholds
    parts = [
        f"""<!doctype html><html lang="ko"><meta charset="utf-8"><title>조건-이력서 유사도 히트맵</title>
<style>body{{font:13px -apple-system,'Apple SD Gothic Neo',sans-serif;margin:24px;background:#faf9f7;color:#1c1c1c}}
table{{border-collapse:collapse;margin:12px 0 28px}}th,td{{border:1px solid #ddd;padding:6px 8px;text-align:center}}
th.req{{text-align:left;max-width:300px;font-weight:500}}td{{font-variant-numeric:tabular-nums;cursor:help}}
.legend span{{display:inline-block;padding:2px 8px;margin-right:6px;border-radius:3px}}h2{{font-size:16px;margin-top:28px}}
.note{{color:#666;max-width:820px;line-height:1.6}}</style>
<h1>조건 × 이력서 유사도</h1>
<p class="note">각 칸은 <b>조건 문장과 가장 비슷한 이력서 청크의 코사인 유사도</b>입니다. 칸에 마우스를 올리면 그 청크가 보입니다.
색은 현재 설정 기준이며, LLM 재판정 구간에 들어간 칸은 옅은 베이지색입니다.</p>
<p class="legend"><span style="background:#1f7a3f;color:#fff">≥ {t.met:.2f} 충족</span>
<span style="background:rgba(230,180,60,0.8)">≥ {t.partial:.2f} 부분 충족</span>
<span style="background:#f3e6cc">≥ {config.llm_judge.band.low:.2f} LLM 재판정 구간</span>
<span style="background:#f8e3e3">그 미만</span>
<span style="background:#eee;color:#888">연차 조건</span></p>"""
    ]
    for job, matrix in jobs_data:
        parts.append(f"<h2>{html.escape(job.title)} · {html.escape(job.company)}</h2><table><tr><th class='req'>조건</th>")
        for candidate, label, _ in matrix:
            parts.append(f"<th>{html.escape(candidate.name)}<br><small>{html.escape(label)}</small></th>")
        parts.append("</tr>")
        for i, requirement in enumerate(job.requirements):
            badge = "필수" if requirement.type.value == "required" else "우대"
            parts.append(f"<tr><th class='req'>[{badge}] {html.escape(requirement.text)}</th>")
            for _, _, column in matrix:
                similarity, evidence = column[i]
                text = "-" if similarity is None else f"{similarity:.2f}"
                parts.append(
                    f'<td style="{_cell_style(similarity, config)}" title="{html.escape(evidence)}">{text}</td>'
                )
            parts.append("</tr>")
        parts.append("</table>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
