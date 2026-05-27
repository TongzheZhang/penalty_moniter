from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agents.audit import AuditEvolutionAgent
from src.agents.context import ContextAgent
from src.agents.decision import DecisionAgent
from src.agents.market_sensor import MarketSensorAgent
from src.agents.paper_execution import PaperExecutionAgent
from src.agents.vision_sensor import VisionSensorAgent
from src.clients.polymarket_client import PolymarketClient, PolymarketClientError
from src.config import Settings
from src.agents.commentary import CommentaryAnalyzer, CommentaryMonitor
from src.live import ClipRecorder, LiveStateProvider, LiveVideoRunner, VideoFrameSource
from src.notifier import Notifier
from src.pipeline import PenaltyResearchPipeline
from src.stream_resolver import DirectUrlResolver, resolve_stream_url
from src.reporting.renderers import format_replay_summary
from src.reporting.html_renderer import render_analysis_report
from src.storage.jsonl_store import RunStore, JsonlStore
from src import tuning, analysis, annotation
from src.batch_replay import run_batch_replay as _batch_replay_core, format_batch_report, save_batch_results
from src.cooldown import CooldownTracker
from src.state_editor import update_state
from src.validation import validate_events, format_validation_report
from src.offline.collector import MatchCollector
from src.offline.preprocessor import MatchPreprocessor
from src.offline.annotator import run_cli_annotator
from src.offline.dataset_builder import DatasetBuilder
from src.offline.trainer import TrainConfig, Trainer
from src.offline.evaluator import Evaluator
from src.online.pipeline import MLOnlinePipeline


ROOT = Path(__file__).parent


class ChineseHelpFormatter(argparse.HelpFormatter):
    SECTION_TITLES = {
        "positional arguments": "位置参数",
        "options": "选项",
        "optional arguments": "可选参数",
        "subcommands": "子命令",
    }

    def start_section(self, heading: str | None) -> None:
        super().start_section(self.SECTION_TITLES.get(heading or "", heading))


class ChineseArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("formatter_class", ChineseHelpFormatter)
        kwargs.setdefault("add_help", False)
        super().__init__(*args, **kwargs)
        self.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")


def build_pipeline(settings: Settings, output_dir: Path | None = None) -> PenaltyResearchPipeline:
    store = RunStore(output_dir or settings.new_run_dir(ROOT))
    cooldown = None
    if settings.cooldown_sec > 0:
        cooldown = CooldownTracker(cooldown_sec=settings.cooldown_sec)
    return PenaltyResearchPipeline(
        settings=settings,
        vision_sensor=VisionSensorAgent(),
        market_sensor=MarketSensorAgent(),
        context_agent=ContextAgent(),
        decision_agent=DecisionAgent(settings),
        paper_execution_agent=PaperExecutionAgent(settings, cooldown=cooldown),
        audit_agent=AuditEvolutionAgent(probability_threshold=settings.probability_threshold),
        store=store,
    )


def run_replay(args: argparse.Namespace) -> int:
    settings = Settings.from_file(ROOT / args.config)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None and not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    pipeline = build_pipeline(settings, output_dir=output_dir)
    summary = pipeline.run_replay(ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input))
    print(format_replay_summary(summary))
    return 0


def run_sample(_: argparse.Namespace) -> int:
    sample_path = ROOT / "data" / "samples" / "replay_events.json"
    print(sample_path.read_text(encoding="utf-8"))
    return 0


def run_tune(args: argparse.Namespace) -> int:
    input_path = ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    thresholds = None
    confidences = None
    if args.thresholds:
        thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    if args.confidences:
        confidences = [float(x.strip()) for x in args.confidences.split(",") if x.strip()]

    settings = Settings.from_file(ROOT / args.config)
    results = tuning.run_grid_search(input_path, thresholds=thresholds, confidences=confidences, base_settings=settings)
    print(tuning.format_tuning_report(results, top_k=args.top_k))

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        tuning.save_tuning_results(results, out_path)
        print(f"\n详细结果已保存至: {out_path}")
    return 0


def run_analyze(args: argparse.Namespace) -> int:
    run_dirs: list[Path] = []
    for d in args.run_dir:
        p = Path(d)
        if not p.is_absolute():
            p = ROOT / p
        run_dirs.append(p)

    results = [analysis.analyze_run(d) for d in run_dirs]

    if args.format == "html":
        html = render_analysis_report(results)
        if args.output:
            out_path = Path(args.output)
            if not out_path.is_absolute():
                out_path = ROOT / out_path
            out_path.write_text(html, encoding="utf-8")
            print(f"HTML 报告已保存至: {out_path}")
        else:
            print(html)
        return 0

    report = analysis.format_analysis_report(results)
    print(report)

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        analysis.save_analysis_report(results, out_path)
        print(f"\nJSON 报告已保存至: {out_path}")
    return 0


def run_review(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir

    candidates = JsonlStore(run_dir / "evolution_candidates.jsonl").read_all()
    audit = JsonlStore(run_dir / "audit.jsonl").read_all()
    audit_by_eid = {a["event_id"]: a for a in audit}

    if args.only_candidates:
        items = candidates
    else:
        items = [a for a in audit if a.get("failure_reason")]

    if not items:
        print("本次运行没有需要审阅的条目。")
        return 0

    print(f"Penalty Monitor - 审阅报告 ({run_dir.name})")
    print("=" * 50)
    for item in items:
        eid = item.get("event_id", "")
        reason = item.get("failure_reason", item.get("candidate_id", ""))
        proposed = item.get("proposed_change", "")
        print(f"\n事件: {eid}")
        print(f"原因: {reason}")
        if proposed:
            print(f"建议: {proposed}")
        if not args.only_candidates:
            pred = audit_by_eid.get(eid, {}).get("prediction", {})
            print(f"预测概率: {pred.get('penalty_probability', 0):.2%} 置信度: {pred.get('confidence', 0):.2%}")
    return 0


def run_annotate(args: argparse.Namespace) -> int:
    source: Path | None = None
    if args.input:
        source = Path(args.input)
        if not source.is_absolute():
            source = ROOT / source
    elif args.run_dir:
        source = Path(args.run_dir)
        if not source.is_absolute():
            source = ROOT / source
    else:
        print("错误: 必须指定 --input 或 --run-dir")
        return 1

    events = annotation.load_unlabeled_events(source)
    if not events:
        print("没有找到未标注的事件。")
        return 0

    if args.batch_mapping:
        mapping_path = Path(args.batch_mapping)
        if not mapping_path.is_absolute():
            mapping_path = ROOT / mapping_path
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        annotated = annotation.batch_annotate(events, mapping)
    else:
        annotated = annotation.interactive_annotate(events)

    if not annotated:
        print("没有产生任何标注结果。")
        return 0

    out_path = Path(args.output) if args.output else ROOT / "data" / "samples" / "annotated_events.json"
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    annotation.save_annotated_events(annotated, out_path)
    print(f"已保存 {len(annotated)} 条标注结果到: {out_path}")
    return 0


def run_update_state(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.is_absolute():
        path = ROOT / path

    signals = {}
    for field in ("box_contact", "fall", "protest", "ref_earpiece", "ref_var_walk", "stoppage"):
        val = getattr(args, field)
        if val is not None:
            signals[f"{field}_score"] = val

    market = {}
    if args.market_id:
        market["market_id"] = args.market_id
    if args.token_id:
        market["token_id"] = args.token_id
    if args.best_ask is not None:
        market["best_ask"] = args.best_ask
    if args.liquidity_usd is not None:
        market["liquidity_usd"] = args.liquidity_usd

    ctx = {}
    if args.home:
        ctx["home"] = args.home
    if args.away:
        ctx["away"] = args.away
    if args.minute is not None:
        ctx["minute"] = args.minute
    if args.attacking_side:
        ctx["attacking_side"] = args.attacking_side

    payload = update_state(path, signals, market_snapshot=market or None, match_context=ctx or None)
    print(f"状态已更新: {path}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run_validate(args: argparse.Namespace) -> int:
    input_path = ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    from src.pipeline import load_replay_events
    events = load_replay_events(input_path)
    issues = validate_events(events)
    print(format_validation_report(issues))

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.write_text(
            json.dumps([i.to_dict() for i in issues], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n验证结果已保存至: {out_path}")

    return 1 if any(i.severity == "error" for i in issues) else 0


def run_batch_replay(args: argparse.Namespace) -> int:
    input_path = ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    base_config = ROOT / args.config
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "data" / "runs" / "batch"
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    overrides = {}
    if args.overrides:
        overrides = json.loads(args.overrides)

    results = _batch_replay_core(input_path, base_config, overrides, output_dir)
    print(format_batch_report(results))

    summary_path = output_dir / "batch_summary.json"
    save_batch_results(results, summary_path)
    print(f"\n批量回测汇总已保存至: {summary_path}")
    return 0


def run_markets(args: argparse.Namespace) -> int:
    settings = Settings.from_file(ROOT / args.config)
    client = PolymarketClient(
        gamma_api_base=settings.gamma_api_base,
        geoblock_url=settings.geoblock_url,
        timeout_sec=settings.timeout_sec,
        max_retries=settings.max_retries,
    )
    try:
        events = client.fetch_events(tag_id=args.tag_id, limit=args.limit)
    except PolymarketClientError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False, indent=2))
        return 1

    rows = [client.to_research_market(event) for event in events]
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def run_geoblock(args: argparse.Namespace) -> int:
    settings = Settings.from_file(ROOT / args.config)
    client = PolymarketClient(
        gamma_api_base=settings.gamma_api_base,
        geoblock_url=settings.geoblock_url,
        timeout_sec=settings.timeout_sec,
        max_retries=settings.max_retries,
    )
    try:
        payload = client.check_geoblock()
    except PolymarketClientError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _resolve_video_source(source: str) -> str:
    """如果是网页 URL，尝试解析为直接流地址。"""
    if not source.startswith(("http://", "https://", "rtmp://", "rtsp://")):
        return source  # 摄像头编号或本地文件
    direct = DirectUrlResolver()
    if direct.can_handle(source):
        return source
    try:
        return resolve_stream_url(source)
    except RuntimeError as exc:
        print(json.dumps(
            {"status": "warning", "message": f"流地址解析失败，将直接尝试原始输入: {exc}"},
            ensure_ascii=False, indent=2,
        ))
        return source


def run_live(args: argparse.Namespace) -> int:
    settings = Settings.from_file(ROOT / args.config)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None and not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    pipeline = build_pipeline(settings, output_dir=output_dir)
    frames_dir = pipeline.store.root / "frames"
    state_file = Path(args.state_file) if args.state_file else None
    if state_file is not None and not state_file.is_absolute():
        state_file = ROOT / state_file

    # 解析视频源（网页 URL -> 直接流地址）
    video_source = _resolve_video_source(args.video_source)

    match_context = {
        "home": args.home,
        "away": args.away,
        "attacking_side": args.attacking_side,
    }
    market_snapshot = {
        "market_id": args.market_id,
        "token_id": args.token_id,
        "best_ask": args.best_ask,
        "liquidity_usd": args.liquidity_usd,
    }
    frame_source = VideoFrameSource(
        source=video_source,
        frames_dir=frames_dir,
        sample_interval_sec=args.sample_interval_sec,
        save_frames=not args.no_save_frames,
    )

    # 解说监控
    commentary_monitor: CommentaryMonitor | None = None
    commentary_mode = args.commentary_mode
    if commentary_mode != "off":
        commentary_file = Path(args.commentary_file) if args.commentary_file else None
        if commentary_file is not None and not commentary_file.is_absolute():
            commentary_file = ROOT / commentary_file
        try:
            commentary_monitor = CommentaryMonitor(
                mode=commentary_mode,
                stream_url=video_source if commentary_mode == "audio" else None,
                commentary_file=commentary_file,
                analyzer=CommentaryAnalyzer(),
                interval_sec=args.commentary_interval_sec,
                whisper_model=args.whisper_model,
                work_dir=pipeline.store.root / "commentary_work",
            )
        except RuntimeError as exc:
            print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False, indent=2))
            return 1

    # 视频片段录制
    clip_recorder: ClipRecorder | None = None
    if args.save_clips:
        clip_recorder = ClipRecorder(
            stream_url=video_source,
            clips_dir=pipeline.store.root / "clips",
            clip_sec=args.clip_sec,
        )

    runner = LiveVideoRunner(
        pipeline=pipeline,
        frame_source=frame_source,
        state_provider=LiveStateProvider(
            path=state_file,
            default_match_context=match_context,
            default_market_snapshot=market_snapshot,
        ),
        match_id=args.match_id,
        source_label=f"live_video:{video_source}",
        print_all=not args.only_alerts,
        notifier=Notifier(settings.notify_config),
        commentary_monitor=commentary_monitor,
        clip_recorder=clip_recorder,
    )
    try:
        summary = runner.run(max_frames=args.max_frames)
    except RuntimeError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def run_resolve_stream(args: argparse.Namespace) -> int:
    try:
        url = resolve_stream_url(args.url)
        print(json.dumps({"status": "ok", "stream_url": url}, ensure_ascii=False, indent=2))
        return 0
    except RuntimeError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False, indent=2))
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = ChineseArgumentParser(description="足球 VAR 点球预判研究系统（仅纸面交易）")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径，默认 config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    replay = sub.add_parser("replay", help="从 JSON/JSONL 事件文件运行离线回放")
    replay.add_argument("--input", required=True, help="输入事件文件路径")
    replay.add_argument("--output-dir", default="", help="输出目录；为空时自动写入 data/runs/<时间>")

    sub.add_parser("sample", help="打印内置 replay 样例")

    live = sub.add_parser("live", help="实时读取视频源并输出点球概率判断")
    live.add_argument("--video-source", required=True, help="视频源：摄像头编号、本地文件、HTTP/RTSP 地址")
    live.add_argument("--match-id", required=True, help="比赛唯一标识，例如 arsenal-chelsea-2026-05-08")
    live.add_argument("--state-file", default="", help="实时状态 JSON 文件；外部视觉模型可持续写入 signals")
    live.add_argument("--sample-interval-sec", type=float, default=1.0, help="抽帧间隔，单位秒")
    live.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧；0 表示持续运行")
    live.add_argument("--output-dir", default="", help="输出目录；为空时自动写入 data/runs/<时间>")
    live.add_argument("--home", default="", help="主队名称，可选")
    live.add_argument("--away", default="", help="客队名称，可选")
    live.add_argument("--attacking-side", default="unknown", help="疑似进攻方：home/away/unknown")
    live.add_argument("--market-id", default="", help="纸面订单使用的市场 ID，可选")
    live.add_argument("--token-id", default="", help="纸面订单使用的 token ID，可选")
    live.add_argument("--best-ask", type=float, default=None, help="纸面参考卖价，可选")
    live.add_argument("--liquidity-usd", type=float, default=None, help="流动性估计，可选")
    live.add_argument("--only-alerts", action="store_true", help="只打印达到阈值的判断")
    live.add_argument("--no-save-frames", action="store_true", help="不保存抽帧图片")
    live.add_argument("--commentary-mode", choices=["off", "file", "audio"], default="off", help="解说监控模式")
    live.add_argument("--commentary-file", default="", help="外部解说文本文件路径（file 模式）")
    live.add_argument("--whisper-model", choices=["tiny", "base", "small"], default="base", help="STT 模型大小（audio 模式）")
    live.add_argument("--commentary-interval-sec", type=float, default=3.0, help="解说采样间隔，单位秒")
    live.add_argument("--save-clips", action="store_true", help="达标时保存视频片段")
    live.add_argument("--clip-sec", type=float, default=10.0, help="视频片段时长，单位秒")

    resolve = sub.add_parser("resolve-stream", help="解析直播网页 URL 为直接流地址")
    resolve.add_argument("--url", required=True, help="直播网页地址")

    markets = sub.add_parser("markets", help="读取公开 Polymarket 赛事事件，仅用于研究")
    markets.add_argument("--tag-id", type=int, required=True, help="Polymarket Gamma tag_id")
    markets.add_argument("--limit", type=int, default=50, help="最多读取多少条事件")

    sub.add_parser("geoblock-check", help="读取当前 IP 的 Polymarket geoblock 状态")

    tune = sub.add_parser("tune", help="对 replay 事件做阈值网格搜索，寻找最优参数")
    tune.add_argument("--input", required=True, help="replay 事件文件路径")
    tune.add_argument("--thresholds", default="", help="概率阈值网格，逗号分隔，如 0.6,0.7,0.8")
    tune.add_argument("--confidences", default="", help="置信度网格，逗号分隔")
    tune.add_argument("--top-k", type=int, default=10, help="Top-K 结果数量")
    tune.add_argument("--output", default="", help="输出 JSON 结果路径")

    analyze = sub.add_parser("analyze", help="分析一次或多次历史运行的指标与信号分布")
    analyze.add_argument("--run-dir", nargs="+", required=True, help="运行目录，可指定多个")
    analyze.add_argument("--output", default="", help="输出报告路径")
    analyze.add_argument("--format", choices=["text", "html"], default="text", help="报告格式")

    review = sub.add_parser("review", help="审阅某次运行的失败案例与进化候选")
    review.add_argument("--run-dir", required=True, help="运行目录")
    review.add_argument("--only-candidates", action="store_true", help="只展示 evolution_candidates")

    annotate = sub.add_parser("annotate", help="对未标注事件进行人工标注")
    annotate.add_argument("--input", default="", help="replay 事件文件路径")
    annotate.add_argument("--run-dir", default="", help="运行目录（读取 evidence.jsonl）")
    annotate.add_argument("--output", default="", help="输出标注后的文件路径")
    annotate.add_argument("--batch-mapping", default="", help="批量标注映射 JSON 文件路径")

    update_state = sub.add_parser("update-state", help="快速更新实时状态文件（边看比赛边写信号）")
    update_state.add_argument("--file", required=True, help="状态文件路径")
    update_state.add_argument("--box-contact", type=float, default=None, help="禁区接触信号 0~1")
    update_state.add_argument("--fall", type=float, default=None, help="倒地信号 0~1")
    update_state.add_argument("--protest", type=float, default=None, help="抗议信号 0~1")
    update_state.add_argument("--ref-earpiece", type=float, default=None, help="裁判耳机信号 0~1")
    update_state.add_argument("--ref-var-walk", type=float, default=None, help="VAR 走向信号 0~1")
    update_state.add_argument("--stoppage", type=float, default=None, help="暂停信号 0~1")
    update_state.add_argument("--market-id", default="", help="市场 ID")
    update_state.add_argument("--token-id", default="", help="Token ID")
    update_state.add_argument("--best-ask", type=float, default=None, help="卖一价")
    update_state.add_argument("--liquidity-usd", type=float, default=None, help="流动性")
    update_state.add_argument("--home", default="", help="主队")
    update_state.add_argument("--away", default="", help="客队")
    update_state.add_argument("--minute", type=int, default=None, help="比赛分钟")
    update_state.add_argument("--attacking-side", default="", help="疑似进攻方 home/away")

    validate = sub.add_parser("validate", help="验证 replay 事件文件的数据质量")
    validate.add_argument("--input", required=True, help="replay 事件文件路径")
    validate.add_argument("--output", default="", help="输出验证结果 JSON 路径")

    batch = sub.add_parser("batch-replay", help="批量回测：对同一数据集运行多组配置变体")
    batch.add_argument("--input", required=True, help="replay 事件文件路径")
    batch.add_argument("--overrides", required=True, help='配置覆盖 JSON，如 {"decision.probability_threshold":[0.7,0.8]}')
    batch.add_argument("--output-dir", default="", help="批量运行根目录")
    batch.add_argument("--config", default="config.yaml", help="基础配置文件路径")

    # 离线数据采集
    collect = sub.add_parser("collect", help="采集公开比赛录像")
    collect.add_argument("--urls", required=True, help="URL列表文件，每行一个")
    collect.add_argument("--output", default="data/matches", help="输出目录")
    collect.add_argument("--dry-run", action="store_true", help="只解析元数据不下载")

    preprocess = sub.add_parser("preprocess", help="预处理比赛录像")
    preprocess.add_argument("--match-dir", required=True, help="比赛目录")
    preprocess.add_argument("--slice-window", type=float, default=10.0, help="切片窗口秒数")
    preprocess.add_argument("--slice-stride", type=float, default=5.0, help="切片步进秒数")
    preprocess.add_argument("--whisper-model", default="base", help="STT模型大小")

    annotate_video = sub.add_parser("annotate-video", help="交互式视频标注工具")
    annotate_video.add_argument("--match-dir", required=True, help="比赛目录")
    annotate_video.add_argument("--transcript", default="", help="转录文件路径")
    annotate_video.add_argument("--labeler", default="anonymous", help="标注者ID")

    build_dataset = sub.add_parser("build-dataset", help="从标注比赛构建训练数据集")
    build_dataset.add_argument("--matches-dir", default="data/matches", help="比赛目录")
    build_dataset.add_argument("--output", default="data/datasets/v1", help="数据集输出目录")
    build_dataset.add_argument("--neg-ratio", type=float, default=4.0, help="负样本比例")

    train = sub.add_parser("train", help="训练点球预测模型")
    train.add_argument("--dataset", default="data/datasets/v1", help="数据集目录")
    train.add_argument("--model-name", default="penalty_predictor_v1", help="模型名称")
    train.add_argument("--device", default="cpu", help="训练设备 cpu/cuda")
    train.add_argument("--epochs", type=int, default=50, help="训练轮数")
    train.add_argument("--batch-size", type=int, default=32, help="批次大小")
    train.add_argument("--lr", type=float, default=1e-4, help="学习率")

    evaluate = sub.add_parser("evaluate", help="评估模型在测试集上的性能")
    evaluate.add_argument("--model", required=True, help="模型文件路径 (.pt 或 .onnx)")
    evaluate.add_argument("--dataset", default="data/datasets/v1", help="数据集目录")
    evaluate.add_argument("--device", default="cpu", help="推理设备")

    watch = sub.add_parser("watch", help="使用ML模型监控直播流")
    watch.add_argument("--stream", required=True, help="直播流地址")
    watch.add_argument("--match-id", required=True, help="比赛ID")
    watch.add_argument("--model", required=True, help="ONNX/PyTorch模型路径")
    watch.add_argument("--threshold", type=float, default=0.75, help="触发阈值")
    watch.add_argument("--max-frames", type=int, default=0, help="最大处理帧数")
    watch.add_argument("--sample-interval", type=float, default=1.0, help="抽帧间隔")

    return parser


def run_collect(args: argparse.Namespace) -> int:
    collector = MatchCollector(output_dir=Path(args.output))
    urls = [line.strip() for line in Path(args.urls).read_text(encoding="utf-8").splitlines() if line.strip()]
    results = collector.batch_download(urls=urls, dry_run=args.dry_run)
    print(json.dumps({"status": "ok", "downloaded": len(results), "dirs": [str(d) for d in results]}, ensure_ascii=False, indent=2))
    return 0


def run_preprocess(args: argparse.Namespace) -> int:
    match_dir = Path(args.match_dir)
    if not match_dir.is_absolute():
        match_dir = ROOT / match_dir
    preprocessor = MatchPreprocessor(match_dir=match_dir, whisper_model=args.whisper_model)
    summary = preprocessor.process_all(
        slice_window=args.slice_window,
        slice_stride=args.slice_stride,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def run_annotate_video(args: argparse.Namespace) -> int:
    match_dir = Path(args.match_dir)
    if not match_dir.is_absolute():
        match_dir = ROOT / match_dir
    transcript_path = Path(args.transcript) if args.transcript else match_dir / "transcript.jsonl"
    return run_cli_annotator(match_dir=match_dir, transcript_path=transcript_path, labeler=args.labeler)


def run_build_dataset(args: argparse.Namespace) -> int:
    matches_dir = Path(args.matches_dir)
    output_dir = Path(args.output)
    if not matches_dir.is_absolute():
        matches_dir = ROOT / matches_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    builder = DatasetBuilder(matches_dir=matches_dir, neg_ratio=args.neg_ratio)
    stats = builder.build(output_dir=output_dir)
    print(json.dumps({"status": "ok", "stats": stats}, ensure_ascii=False, indent=2))
    return 0


def run_train(args: argparse.Namespace) -> int:
    dataset_dir = ROOT / args.dataset
    config = TrainConfig(
        dataset_dir=str(dataset_dir),
        model_name=args.model_name,
        device=args.device,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    trainer = Trainer(config)
    result = trainer.train()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    model_path = Path(args.model)
    dataset_dir = ROOT / args.dataset
    evaluator = Evaluator(model_path=model_path, device=args.device)
    metrics = evaluator.evaluate(dataset_dir=dataset_dir)
    print(json.dumps({"status": "ok", "metrics": metrics.to_dict()}, ensure_ascii=False, indent=2))
    return 0


def run_watch(args: argparse.Namespace) -> int:
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    pipeline = MLOnlinePipeline(
        model_path=model_path,
        probability_threshold=args.threshold,
    )
    try:
        summary = pipeline.watch_stream(
            stream_url=args.stream,
            match_id=args.match_id,
            max_frames=args.max_frames,
            sample_interval_sec=args.sample_interval,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except RuntimeError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "replay":
        return run_replay(args)
    if args.command == "sample":
        return run_sample(args)
    if args.command == "live":
        return run_live(args)
    if args.command == "resolve-stream":
        return run_resolve_stream(args)
    if args.command == "markets":
        return run_markets(args)
    if args.command == "geoblock-check":
        return run_geoblock(args)
    if args.command == "tune":
        return run_tune(args)
    if args.command == "analyze":
        return run_analyze(args)
    if args.command == "review":
        return run_review(args)
    if args.command == "annotate":
        return run_annotate(args)
    if args.command == "update-state":
        return run_update_state(args)
    if args.command == "validate":
        return run_validate(args)
    if args.command == "batch-replay":
        return run_batch_replay(args)
    if args.command == "collect":
        return run_collect(args)
    if args.command == "preprocess":
        return run_preprocess(args)
    if args.command == "annotate-video":
        return run_annotate_video(args)
    if args.command == "build-dataset":
        return run_build_dataset(args)
    if args.command == "train":
        return run_train(args)
    if args.command == "evaluate":
        return run_evaluate(args)
    if args.command == "watch":
        return run_watch(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
