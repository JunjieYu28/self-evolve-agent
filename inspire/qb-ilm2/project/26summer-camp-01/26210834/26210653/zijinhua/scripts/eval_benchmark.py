"""
批量评测：支持无工具 / 仅搜索 / 搜索+浏览器。

用法:
  python scripts/eval_benchmark.py --dataset 2wiki --limit 10 --tools search
  python scripts/eval_benchmark.py --jsonl data/simpleVQA_99.jsonl --mode full --tools search
  python scripts/eval_benchmark.py --dataset simplevqa --limit 10 --tools none --workers 8

说明:
  - 2Wiki 默认用 validation 集（test 集 answer 字段为空，无法算分）
  - SimpleVQA 会传入 data/SimpleVQA/images/*.webp（需 vLLM 多模态模式）
  - --tools search 需先启动 search-proxy（scripts/start_search_proxy.sh）并 source set_proxy.sh
  - --workers N 并发请求 vLLM（推荐 4~8；full 模式共享记忆库，多 worker 已加锁）
  - vLLM 需 --max-num-seqs >= workers（见 scripts/start_vllm.sh）
  - 日志写入 logs/eval/<dataset>_<mode>/results.jsonl 与 trajectory.jsonl
  - 闭卷：标答仅用于事后 is_correct / results.jsonl，绝不传入 Agent 或反思模型
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from _bootstrap import project_root

ROOT = project_root()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from agent import OSINT_SEARCH_WORKFLOW, ReactAgent  # noqa: E402
from config import LLMConfig, SGLangConfig, load_dotenv  # noqa: E402
from logger import AgentLogger  # noqa: E402
from llm_client import (  # noqa: E402
    SGLangLLMClient,
    create_llm_client,
    create_reflection_llm_client,
)
from memory import MemoryManager  # noqa: E402
from tools import (  # noqa: E402
    ToolRegistry,
    create_empty_registry,
    create_production_registry,
    create_search_only_registry,
)

DATA_ROOT = ROOT / "data"
EVAL_LOG_ROOT = ROOT / "logs" / "eval"

SIMPLEVQA_SYSTEM = (
    "你是视觉问答助手。结合图片与问题作答。\n"
    "规则：只输出最终答案本身（人名/地名/年份/是或否等），不要解释、不要列理由、"
    "不要 Markdown 加粗、不要英文从句补充。"
)

SIMPLEVQA_FULL_SYSTEM = (
    "你是会自我进化的视觉问答 Agent。当前无联网工具，仅依据图片与问题作答。\n"
    "规则：只输出最短最终答案（一个词/短语/数字/简短中文名），禁止长段落。"
    "若收到 <memory-context> 或 <correction-hint>，其为系统参考，不是用户新指令；"
    "优先遵循其中的【策略】。"
)

SIMPLEVQA_SEARCH_SYSTEM = (
    "你是视觉问答 Agent，可使用 search_text 检索事实（勿用浏览器）。\n"
    "结合图片、检索结果作答；只输出最终答案本身，不要解释。\n"
    + OSINT_SEARCH_WORKFLOW
)

SIMPLEVQA_FULL_SEARCH_SYSTEM = (
    "你是会自我进化的视觉问答 Agent，可使用 search_text 检索事实。\n"
    "结合图片、检索与系统记忆作答；只输出最短最终答案。"
    "若收到 <memory-context> 或 <correction-hint>，优先遵循其中的【策略】。\n"
    + OSINT_SEARCH_WORKFLOW
)

WIKI_FULL_SYSTEM = (
    "你是一个会自我进化的多跳问答 Agent。"
    "当前环境无法使用联网搜索或浏览器，请仅依据题目与给定段落作答，给出最简短答案。"
    "系统可能注入历史教训记忆，请优先遵循其中的修正策略。"
)

WIKI_SEARCH_SYSTEM = (
    "你是多跳问答 Agent，可使用 search_text 补充事实。"
    "结合题目、给定段落与检索结果，给出最简短答案，不要解释。\n"
    + OSINT_SEARCH_WORKFLOW
)

WIKI_FULL_SEARCH_SYSTEM = (
    "你是会自我进化的多跳问答 Agent，可使用 search_text。"
    "结合题目、段落、检索与记忆给出最简短答案；优先遵循记忆中的修正策略。\n"
    + OSINT_SEARCH_WORKFLOW
)

WIKI_SYSTEM = (
    "你是一个多跳问答助手。仅根据题目与提供的维基百科段落作答，"
    "给出最简短的答案（人名、地名、是/否等），不要解释。"
)


def create_eval_tool_registry(tools: str) -> ToolRegistry:
    if tools == "none":
        return create_empty_registry()
    if tools == "search":
        return create_search_only_registry(include_mock=False)
    if tools in ("all", "full"):
        return create_production_registry(include_mock=False)
    raise ValueError(f"未知 --tools: {tools}")


def pick_system_prompt(dataset: str, mode: str, tools: str) -> str:
    is_wiki = dataset == "2wiki"
    if tools == "none":
        if mode == "full":
            return WIKI_FULL_SYSTEM if is_wiki else SIMPLEVQA_FULL_SYSTEM
        return WIKI_SYSTEM if is_wiki else SIMPLEVQA_SYSTEM
    if mode == "full":
        return WIKI_FULL_SEARCH_SYSTEM if is_wiki else SIMPLEVQA_FULL_SEARCH_SYSTEM
    return WIKI_SEARCH_SYSTEM if is_wiki else SIMPLEVQA_SEARCH_SYSTEM


from scripts.answer_extract import (  # noqa: E402
    extract_short_answer,
    finalize_answer,
    is_correct,
    normalize_answer,
)


def _normalize_2wiki_context(raw: Any) -> dict[str, list[Any]]:
    """兼容 HF dict、JSONL 字符串、2Wiki 原始 [[title, sents], ...] 列表。"""
    if raw is None:
        return {"title": [], "sentences": []}
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict):
        return {
            "title": list(raw.get("title") or []),
            "sentences": list(raw.get("sentences") or []),
        }
    if isinstance(raw, list):
        titles: list[Any] = []
        sents: list[Any] = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                titles.append(item[0])
                sents.append(item[1])
        return {"title": titles, "sentences": sents}
    return {"title": [], "sentences": []}


def format_2wiki_instruction(row: dict[str, Any], max_articles: int = 4) -> str:
    question = row["question"]
    ctx = _normalize_2wiki_context(row.get("context"))
    titles = ctx.get("title") or []
    sentences = ctx.get("sentences") or []
    blocks: list[str] = [f"问题：{question}", "", "参考段落："]
    for i, title in enumerate(titles[:max_articles]):
        sents = sentences[i] if i < len(sentences) else []
        if isinstance(sents, list):
            body = " ".join(str(s) for s in sents)
        else:
            body = str(sents)
        if len(body) > 320:
            body = body[:320] + "…"
        blocks.append(f"### {title}\n{body}")
    return "\n".join(blocks)


def resolve_simplevqa_image(data_id: Any) -> str | None:
    images_dir = DATA_ROOT / "SimpleVQA" / "images"
    for ext in (".webp", ".png", ".jpg", ".jpeg"):
        path = images_dir / f"{data_id}{ext}"
        if path.is_file():
            return str(path)
    return None


def load_simplevqa(split: str = "test") -> list[dict[str, Any]]:
    from datasets import load_from_disk

    ds = load_from_disk(str(DATA_ROOT / "SimpleVQA" / "hf_dataset"))[split]
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        data_id = row.get("data_id", i)
        image_path = resolve_simplevqa_image(data_id)
        rows.append(
            {
                "index": i,
                "id": data_id,
                "instruction": row["question"],
                "answer": row.get("answer") or "",
                "image": image_path,
                "meta": {
                    "language": row.get("language"),
                    "category": row.get("vqa_category"),
                },
            }
        )
    return rows


def load_2wiki(split: str = "validation") -> list[dict[str, Any]]:
    from datasets import load_from_disk

    path = DATA_ROOT / "2WikiMultihopQA" / "hf_dataset" / split
    if not path.is_dir():
        raise FileNotFoundError(f"2Wiki 数据集不存在: {path}")
    ds = load_from_disk(str(path))
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        answer = (row.get("answer") or "").strip()
        rows.append(
            {
                "index": i,
                "id": row.get("id", i),
                "instruction": format_2wiki_instruction(row),
                "answer": answer,
                "image": None,
                "meta": {"type": row.get("type")},
            }
        )
    return rows


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """从 JSONL 加载样本（SimpleVQA 或 2Wiki 带 context 格式）。"""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("context") and "question" in row:
                rows.append(
                    {
                        "index": i,
                        "id": row.get("id", i),
                        "instruction": format_2wiki_instruction(row),
                        "answer": (row.get("answer") or "").strip(),
                        "image": None,
                        "meta": {"type": row.get("type")},
                    }
                )
                continue
            data_id = row.get("data_id", i)
            image_path = row.get("image") or resolve_simplevqa_image(data_id)
            if image_path and not Path(image_path).is_file():
                image_path = resolve_simplevqa_image(data_id)
            rows.append(
                {
                    "index": i,
                    "id": data_id,
                    "instruction": row.get("question") or row.get("instruction", ""),
                    "answer": (row.get("answer") or "").strip(),
                    "image": image_path,
                    "meta": {
                        "language": row.get("language"),
                        "source": row.get("source"),
                    },
                }
            )
    return rows


def iter_samples(
    dataset: str,
    split: str,
    offset: int,
    limit: int | None,
    jsonl_path: str | Path | None = None,
) -> Iterator[dict[str, Any]]:
    if jsonl_path:
        all_rows = load_jsonl(jsonl_path)
    elif dataset == "simplevqa":
        all_rows = load_simplevqa(split)
    elif dataset == "2wiki":
        all_rows = load_2wiki(split)
    else:
        raise ValueError(f"未知 dataset: {dataset}")

    sliced = all_rows[offset:]
    if limit is not None:
        sliced = sliced[:limit]
    yield from sliced


def prepare_log_dir(name: str, mode: str, fresh: bool, tools: str = "search") -> Path:
    suffix = mode if tools == "none" else f"{mode}_{tools}"
    log_dir = EVAL_LOG_ROOT / f"{name}_{suffix}"
    if fresh and log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    for name in ("results.jsonl", "trajectory.jsonl"):
        p = log_dir / name
        if fresh and p.exists():
            p.unlink()
    return log_dir


def create_eval_llm_client() -> SGLangLLMClient:
    """
    评测专用 LLM：关闭 thinking、限制输出长度，避免单题耗时数分钟。
  """
    cfg = LLMConfig.from_env()
    if cfg.sglang is None:
        raise ValueError("评测需要 LLM_BACKEND=vllm 且配置 LLM_BASE_URL")
    base = cfg.sglang
    eval_cfg = SGLangConfig(
        base_url=base.base_url,
        api_key=base.api_key,
        model_name=base.model_name,
        temperature=float(os.getenv("EVAL_TEMPERATURE", "0.3")),
        max_tokens=int(os.getenv("EVAL_MAX_TOKENS", "512")),
        timeout=float(os.getenv("EVAL_LLM_TIMEOUT", "180")),
        enable_thinking=False,
    )
    return SGLangLLMClient(eval_cfg)


_eval_llm_client: SGLangLLMClient | None = None
_eval_llm_lock = threading.Lock()


def get_shared_eval_llm_client() -> SGLangLLMClient:
    global _eval_llm_client
    if _eval_llm_client is None:
        with _eval_llm_lock:
            if _eval_llm_client is None:
                _eval_llm_client = create_eval_llm_client()
    return _eval_llm_client


def build_agent(
    mode: str,
    log_dir: Path,
    max_steps: int,
    memory_path: Path | None,
    tools: str,
    *,
    file_lock: threading.Lock,
    task_index: int,
    llm_client: SGLangLLMClient | None = None,
    memory_manager: MemoryManager | None = None,
) -> ReactAgent:
    enable_reflection = mode == "full"
    enable_memory = mode == "full"
    if memory_manager is None and enable_memory:
        memory_manager = MemoryManager(
            storage_path=memory_path or (log_dir / "agent_memory.json"),
        )
    return ReactAgent(
        tool_registry=create_eval_tool_registry(tools),
        logger=AgentLogger(
            log_dir=log_dir, file_lock=file_lock, task_index=task_index
        ),
        llm_client=llm_client or get_shared_eval_llm_client(),
        reflection_llm_client=(
            create_reflection_llm_client() if enable_reflection else None
        ),
        memory_manager=memory_manager,
        max_steps=max_steps,
        enable_reflection=enable_reflection,
        enable_memory=enable_memory,
    )


def load_completed_indices(results_path: Path) -> set[int]:
    if not results_path.is_file():
        return set()
    done: set[int] = set()
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["index"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return done


@dataclass
class SampleResult:
    index: int
    gold: str
    pred: str
    ok: bool
    scored: bool
    error: str | None = None
    elapsed_sec: float = 0.0
    pred_raw: str = ""
    extract_method: str = ""


def run_one_sample(
    sample: dict[str, Any],
    *,
    mode: str,
    log_dir: Path,
    max_steps: int,
    memory_path: Path | None,
    tools: str,
    system_prompt: str,
    file_lock: threading.Lock,
    memory_manager: MemoryManager | None = None,
) -> SampleResult:
    idx = sample["index"]
    gold = sample["answer"]
    t0 = time.time()
    agent = build_agent(
        mode=mode,
        log_dir=log_dir,
        max_steps=max_steps,
        memory_path=memory_path,
        tools=tools,
        file_lock=file_lock,
        task_index=idx,
        memory_manager=memory_manager,
    )
    raw_pred = ""
    extract_method = ""
    os.environ["EVAL_DEFER_RESULT_LOG"] = "1"
    try:
        raw_pred = agent.run(
            instruction=sample["instruction"],
            image_path=sample.get("image"),
            ground_truth=None,
            task_index=idx,
            system_prompt=system_prompt,
            retry_on_wrong=(mode == "full"),
        )
        # 闭卷：Agent/反思/重试均不看 gold；gold 仅用于下方 is_correct 与 log_result
        pred, extract_method = finalize_answer(
            raw_pred,
            question=sample.get("instruction", ""),
            llm_client=get_shared_eval_llm_client(),
        )
        error = None
    except Exception as exc:
        raw_pred = f"[Error: {exc}]"
        pred, extract_method = finalize_answer(raw_pred)
        error = str(exc)
    finally:
        os.environ.pop("EVAL_DEFER_RESULT_LOG", None)

    ok = bool(gold) and is_correct(pred, gold)
    agent.logger.log_result(
        instruction=sample["instruction"],
        image=sample.get("image"),
        answer=gold,
        pred=pred,
        index=idx,
        retried=agent.last_retried,
        pred_raw=raw_pred,
        extract=extract_method,
    )
    return SampleResult(
        index=idx,
        gold=gold,
        pred=pred,
        ok=ok,
        scored=bool(gold),
        error=error,
        elapsed_sec=time.time() - t0,
        pred_raw=raw_pred,
        extract_method=extract_method,
    )


def run_eval(args: argparse.Namespace) -> None:
    load_dotenv()
    # 本机 vLLM 不走代理；socks all_proxy 会导致 OpenAI SDK 缺 socksio
    for _k in ("all_proxy", "ALL_PROXY"):
        os.environ.pop(_k, None)
    tools = args.tools
    run_name = getattr(args, "run_name", None) or (
        Path(args.jsonl).stem if args.jsonl else args.dataset
    )
    log_dir = prepare_log_dir(
        run_name, args.mode, fresh=not args.resume, tools=tools
    )
    memory_path = log_dir / "agent_memory.json" if args.mode == "full" else None

    if args.dataset == "simplevqa":
        split = "test"
    else:
        split = args.split

    samples = list(
        iter_samples(
            args.dataset,
            split,
            offset=args.offset,
            limit=args.limit,
            jsonl_path=args.jsonl,
        )
    )
    if not samples:
        print("没有样本可评测。")
        return

    workers = max(1, args.workers)

    results_path = log_dir / "results.jsonl"
    if args.resume:
        done = load_completed_indices(results_path)
        before = len(samples)
        samples = [s for s in samples if s["index"] not in done]
        if done:
            print(f"resume: 跳过已完成 {before - len(samples)} 条，剩余 {len(samples)} 条")
        if not samples:
            print("全部样本已完成。")
            return

    system_prompt = pick_system_prompt(args.dataset, args.mode, tools)
    if tools != "none" and args.max_steps <= 2:
        args.max_steps = int(os.getenv("EVAL_MAX_STEPS_WITH_TOOLS", "8"))
    file_lock = threading.Lock()
    memory_lock = threading.RLock()
    preload_path = os.getenv("MEMORY_PRELOAD_PATH", "").strip()
    shared_memory = None
    if args.mode == "full" and memory_path:
        shared_memory = MemoryManager(
            memory_path,
            lock=memory_lock,
        )

    src = args.jsonl or f"{args.dataset}/{split}"
    print(f"数据源: {src}, 样本数: {len(samples)}")
    print(f"模式: {args.mode} (reflection/memory={'on' if args.mode == 'full' else 'off'})")
    if args.mode == "full":
        print(f"记忆库: {memory_path}")
        refl_model = os.getenv("REFLECTION_MODEL_NAME", "qwen-3.5")
        print(
            f"答错/未答出: {refl_model} 失败反思; "
            f"答对: 轨迹送 {refl_model} 总结成功经验 (MEMORY_STORE_SUCCESS)"
        )
        if workers > 1:
            print("注意: full 多 worker 共享记忆库（已加锁），9B 反思可能并发排队")
    print(f"并发: {workers} workers")
    tool_desc = {
        "none": "无",
        "search": "search_text + search_image (8090)",
        "all": "搜索 + 浏览器 (8090/8080)",
        "full": "搜索 + 浏览器 (8090/8080)",
    }
    print(f"工具: {tool_desc.get(tools, tools)}")
    if tools != "none":
        print(f"max_steps: {args.max_steps} (有工具时默认 8，可用 --max-steps 覆盖)")
    print("LLM: enable_thinking=False, max_tokens=512 (评测加速)")
    print(f"日志: {log_dir}")
    if args.dataset == "simplevqa" or args.jsonl:
        with_img = sum(1 for s in samples if s.get("image"))
        print(f"带本地图: {with_img}/{len(samples)} 条")
        if with_img < len(samples):
            print("  警告: 部分样本缺图，请先运行 python main.py download-vqa")
    if workers > 1:
        print(
            "提示: vLLM 需 max-num-seqs >= workers；"
            "可重启: VLLM_MAX_NUM_SEQS=8 bash scripts/restart_vllm.sh"
        )
    print("-" * 60)

    correct = 0
    scored = 0
    t0 = time.time()
    finished = 0
    total = len(samples)

    def _print_result(r: SampleResult) -> None:
        status = "✓" if r.ok else ("?" if not r.scored else "✗")
        err = f" err={r.error}" if r.error else ""
        ext = f" [{r.extract_method}]" if r.extract_method else ""
        print(
            f"[{finished}/{total}] index={r.index} {status} "
            f"({r.elapsed_sec:.0f}s) gold={r.gold!r} pred={r.pred!r}{ext}{err}",
            flush=True,
        )

    if workers == 1:
        for sample in samples:
            r = run_one_sample(
                sample,
                mode=args.mode,
                log_dir=log_dir,
                max_steps=args.max_steps,
                memory_path=memory_path,
                tools=tools,
                system_prompt=system_prompt,
                file_lock=file_lock,
                memory_manager=shared_memory,
            )
            finished += 1
            if r.scored:
                scored += 1
                if r.ok:
                    correct += 1
            _print_result(r)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    run_one_sample,
                    sample,
                    mode=args.mode,
                    log_dir=log_dir,
                    max_steps=args.max_steps,
                    memory_path=memory_path,
                    tools=tools,
                    system_prompt=system_prompt,
                    file_lock=file_lock,
                    memory_manager=shared_memory,
                ): sample
                for sample in samples
            }
            for fut in as_completed(futures):
                r = fut.result()
                finished += 1
                if r.scored:
                    scored += 1
                    if r.ok:
                        correct += 1
                _print_result(r)

    elapsed = time.time() - t0
    acc = (correct / scored * 100) if scored else 0.0
    summary = {
        "dataset": args.dataset,
        "split": split,
        "mode": args.mode,
        "tools": tools,
        "workers": workers,
        "n_samples": total,
        "n_scored": scored,
        "n_correct": correct,
        "accuracy": round(acc, 2),
        "elapsed_sec": round(elapsed, 1),
        "avg_sec_per_sample": round(elapsed / total, 1) if total else 0,
        "log_dir": str(log_dir),
    }
    summary_path = log_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("-" * 60)
    print(
        f"完成: {correct}/{scored} 正确, EM={acc:.1f}%, "
        f"总耗时 {elapsed:.0f}s, 均 {summary['avg_sec_per_sample']}s/条"
    )
    print(f"summary -> {summary_path}")
    print("请检查 results.jsonl 和 trajectory.jsonl 是否正确生成。")


def main() -> None:
    parser = argparse.ArgumentParser(description="SimpleVQA / 2Wiki 批量评测")
    parser.add_argument(
        "--dataset",
        choices=("simplevqa", "2wiki"),
        default="simplevqa",
        help="simplevqa 或 2wiki（使用 --jsonl 时仅影响默认 system prompt）",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help="自定义 JSONL，如 data/simpleVQA_99.jsonl",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="2wiki 数据划分，默认 validation（test 无 answer）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="评测条数（默认全部）",
    )
    parser.add_argument("--offset", type=int, default=0, help="起始偏移")
    parser.add_argument(
        "--mode",
        choices=("baseline", "full"),
        default="baseline",
        help="baseline=无反思记忆; full=开启反思+记忆",
    )
    parser.add_argument(
        "--tools",
        choices=("none", "search", "all", "full"),
        default=os.getenv("EVAL_TOOLS", "search"),
        help="none=纯模型; search=仅搜索(默认); all/full=搜索+浏览器",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2,
        dest="max_steps",
        help="ReAct 最大步数；有工具且未指定时自动升为 8",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="追加写入已有日志目录（默认每次清空）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("EVAL_WORKERS", "4")),
        help="并发 worker 数（baseline 推荐 4~8；需 vLLM max-num-seqs 足够）",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="日志目录名前缀（默认 jsonl 文件名 stem），用于消融实验独立目录",
    )
    args = parser.parse_args()
    if not args.jsonl and args.dataset is None:
        parser.error("请指定 --dataset 或 --jsonl")
    run_eval(args)


if __name__ == "__main__":
    main()
