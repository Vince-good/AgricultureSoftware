"""命令行入口 `heyan`（pyproject 里 `[project.scripts] heyan = "heyan.cli:main"`）。

一条命令对应方案文档里一个可交付动作：

    heyan build        跑完整流水线（微调→蒸馏→剪枝→导出→量化→预算校验→打包）
    heyan demo-data    生成合成演示数据集（无真实照片时验证流水线用）
    heyan ingest       导入团队田间照片（labels.csv）
    heyan recognize    单张图离线识别（不联网、不启动服务）
    heyan info         查看模型包元信息与预算判定
    heyan benchmark    对已有 bundle 复测体积/延迟/内存
    heyan voicepack    离线预渲染语音包
    heyan serve        启动本地 Web 界面（拍照→识别→语音播报）
                       加 --lan 让同一局域网的其他电脑/手机也能打开
    heyan tunnel       内网穿透：起服务 + 起 cloudflared/ngrok/cpolar，给出一个公网链接
    heyan export       导出识别记录（JSON/CSV，可写 U 盘）
    heyan outbox       查看/冲刷对接发件箱（保险/补贴/农资）
    heyan schema       写出全部 JSON Schema 供对接方审阅
    heyan usability    运行可用性量表（SUS + TAM）并存档

所有子命令都做了惰性 import：`heyan build` 不需要装 flask，
`heyan serve` 不需要装 torch —— 边缘设备上只装 numpy+pillow 也能跑识别。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _resolve_bundle(args) -> Path:
    bundle = Path(args.bundle).expanduser() if getattr(args, "bundle", None) else None
    if bundle is None:
        from .config import PATHS

        candidates = sorted(PATHS.bundles.glob("*"), key=lambda p: p.stat().st_mtime,
                            reverse=True)
        candidates = [c for c in candidates if (c / "manifest.json").exists()]
        if not candidates:
            raise SystemExit("没有找到模型包：先运行 `heyan build` 或用 --bundle 指定")
        bundle = candidates[0]
        print(f"[heyan] 使用最新模型包 {bundle}", file=sys.stderr)
    return bundle


# --------------------------------------------------------------------------
# 子命令
# --------------------------------------------------------------------------

def cmd_build(args) -> int:
    from .train.pipeline import PipelineConfig, run_pipeline

    cfg = PipelineConfig(
        data_dir=args.data_dir, labels_csv=args.labels_csv, image_root=args.image_root,
        arch=args.arch, teacher_arch=args.teacher_arch,
        model_id=args.model_id, version=args.version,
        epochs=args.epochs, teacher_epochs=args.teacher_epochs,
        recover_epochs=args.recover_epochs, batch_size=args.batch_size,
        strategy=args.strategy, unfreeze_blocks=args.unfreeze_blocks,
        seed=args.seed, threads=args.threads, patience=args.patience,
        distill=not args.no_distill, do_prune=not args.no_prune,
        quantize=not args.no_quantize, per_channel=not args.per_tensor,
        calibration_samples=args.calibration_samples,
        qat=not args.no_qat, qat_epochs=args.qat_epochs, qat_lr=args.qat_lr,
        out_root=args.out, region=args.region, notes=args.notes,
        pack_zip=args.pack_zip, benchmark_rounds=args.benchmark_rounds,
        benchmark_threads=args.benchmark_threads,
        build_voicepack=not args.no_voicepack,
        voice_languages=tuple(args.voice_langs.split(",")) if args.voice_langs else ("zh",),
        device_ram_mb=args.device_ram_mb, device_cost_cny=args.device_cost_cny,
    )
    report = run_pipeline(cfg)
    print()
    print(f"[heyan] bundle  -> {report['bundle_dir']}")
    print(f"[heyan] 预算达标 -> {'是' if report['budget_ok'] else '否（见 benchmark.txt）'}")
    return 0 if report["budget_ok"] else 2


def cmd_demo_data(args) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.make_demo_dataset import build_dataset

    report = build_dataset(args.out, per_class=args.per_class, seed=args.seed,
                           size=args.size, write_labels_csv=args.labels_csv,
                           progress=not args.quiet)
    _print_json(report.to_dict())
    return 0


def cmd_ingest(args) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.ingest_field_samples import ingest

    report = ingest(labels_csv=args.labels_csv, image_root=args.image_root,
                    out_dir=args.out, min_per_class=args.min_per_class)
    _print_json(report)
    return 0


def cmd_recognize(args) -> int:
    from .core import RecognitionEngine

    bundle = _resolve_bundle(args)
    with RecognitionEngine(bundle, backend=args.backend, language=args.lang) as engine:
        result = engine.recognize(args.image, language=args.lang)
    payload = result.to_dict()
    if args.brief:
        print(result.summary_line())
        print(f"  播报：{result.advice.voice}")
        for act in result.advice.actions:
            print(f"  - {act}")
    else:
        _print_json(payload)
    return 0


def cmd_info(args) -> int:
    from .core import ModelBundle, RecognitionEngine

    bundle = ModelBundle.load(_resolve_bundle(args))
    info: Dict[str, Any] = {
        "bundle_dir": str(bundle.root),
        "manifest": bundle.manifest.to_dict(),
        "size": {
            "model_mb": bundle.model_size_mb(),
            "bundle_mb": bundle.size_mb(),
            "size_budget": bundle.check_size_budget(),
        },
        "available_kinds": bundle.available_kinds(),
    }
    if args.probe:
        with RecognitionEngine(bundle) as engine:
            info["engine"] = engine.info()
    _print_json(info)
    return 0


def cmd_benchmark(args) -> int:
    from .eval.benchmark import benchmark_bundle, format_report, save_report

    bundle = _resolve_bundle(args)
    images = [Path(p) for p in args.images] if args.images else None
    report = benchmark_bundle(bundle, images=images, rounds=args.rounds,
                              threads=args.threads, device_ram_mb=args.device_ram_mb,
                              device_cost_cny=args.device_cost_cny)
    print(format_report(report))
    if args.save:
        path = save_report(report, Path(args.save))
        print(f"[heyan] 报告 -> {path}")
    return 0 if report["budget_ok"] else 2


def cmd_voicepack(args) -> int:
    from .tts.voicepack import build_voicepack

    langs = [c.strip() for c in args.langs.split(",") if c.strip()]
    report = build_voicepack(langs, args.out, rate=args.rate, force=args.force)
    _print_json(report)
    if args.into_bundle:
        from .tts.voicepack import copy_into_bundle

        dst = copy_into_bundle(args.out, _resolve_bundle(args))
        print(f"[heyan] 语音包已拷入 bundle -> {dst}")
    return 0


def cmd_serve(args) -> int:
    from .server import guard
    from .server.app import create_app, run_server

    host = "0.0.0.0" if getattr(args, "lan", False) else args.host
    public = bool(getattr(args, "public", False) or getattr(args, "tunnel", False))
    if getattr(args, "tunnel", False):
        # 隧道客户端从本机连进来，没必要顺手把网卡也敞开
        host = "127.0.0.1"
    token = getattr(args, "access_token", None) or os.environ.get("HEYAN_ACCESS_TOKEN")
    if public and not token:
        token = guard.generate_token()
        print("[heyan] 没给 --access-token，已自动生成一个（只在这次运行有效）")
    try:
        app = create_app(bundle=args.bundle, language=args.lang, host=host, port=args.port,
                         access_token=token, public=public)
        run_server(app, host=host, port=args.port, debug=args.debug,
                   ssl_cert=getattr(args, "ssl_cert", None),
                   ssl_key=getattr(args, "ssl_key", None),
                   access_token=token, public=public)
    except ValueError as exc:
        print(f"[heyan] {exc}")
        return 2
    return 0


def cmd_tunnel(args) -> int:
    from .server import tunnel

    return tunnel.run(port=args.port, client=args.client, token=args.access_token,
                      subdomain=args.subdomain, region=args.region,
                      bundle=args.bundle, lang=args.lang, timeout=args.timeout)


def cmd_export(args) -> int:
    from .data import Exporter, RecordStore

    store = RecordStore(args.db) if args.db else RecordStore()
    exporter = Exporter(store, out_root=args.out)
    filters = {}
    if args.crop:
        filters["crop"] = args.crop
    if args.class_id:
        filters["class_id"] = args.class_id
    if args.since:
        filters["since"] = args.since
    if args.until:
        filters["until"] = args.until
    result = exporter.export(fmt=args.fmt, purpose=args.purpose, channel=args.channel,
                             filters=filters or None, anonymize=args.anonymize,
                             usb_target=args.usb_target)
    _print_json(result.to_dict())
    return 0 if result.ok else 2


def cmd_outbox(args) -> int:
    from .data import registry

    if args.action == "status":
        _print_json(registry.status())
    elif args.action == "describe":
        _print_json(registry.describe())
    elif args.action == "schemas":
        out = Path(args.out)
        paths = registry.write_schemas(out)
        for p in paths:
            print(p)
    elif args.action == "flush":
        from .data import Outbox

        box = Outbox()
        moved = box.flush(Path(args.target), channel=args.channel)
        _print_json({"flushed": [e.to_dict() for e in moved], "target": str(args.target)})
    elif args.action == "dispatch":
        from .data import RecordStore

        store = RecordStore(args.db) if args.db else RecordStore()
        names = [n.strip() for n in args.names.split(",")] if args.names else None
        results = registry.dispatch_all(store, names)
        _print_json([r.to_dict() for r in results])
    return 0


def cmd_schema(args) -> int:
    from .data import registry, write_schema
    from .data.schema import EXPORT_SCHEMA, RECORD_SCHEMA

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = [write_schema(RECORD_SCHEMA, out / "record.schema.json"),
             write_schema(EXPORT_SCHEMA, out / "export.schema.json")]
    paths += registry.write_schemas(out)
    for p in paths:
        print(p)
    return 0


def cmd_usability(args) -> int:
    from .eval.usability import run_console_session

    report = run_console_session(language=args.lang, out=args.out,
                                 participant_id=args.participant)
    _print_json(report)
    return 0


# --------------------------------------------------------------------------
# 参数解析
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="heyan",
        description="禾眼 HeYan —— 面向小农户的离线作物胁迫视觉识别系统")
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="<command>")

    p = sub.add_parser("build", help="跑完整训练+压缩+量化+打包流水线")
    p.add_argument("--data-dir", help="ImageFolder 布局的数据集目录")
    p.add_argument("--labels-csv", help="filename,class_id 两列的标签清单")
    p.add_argument("--image-root", help="与 labels-csv 配套的图片根目录")
    p.add_argument("--arch", default="mobilenet_v3_small")
    p.add_argument("--teacher-arch", default="mobilenet_v3_large")
    p.add_argument("--model-id", default="heyan-mnv3s-int8")
    p.add_argument("--version", default="1.0.0")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--teacher-epochs", type=int, default=14)
    p.add_argument("--recover-epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--strategy", default="partial", choices=["full", "partial", "head"])
    p.add_argument("--unfreeze-blocks", type=int, default=4)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--no-distill", action="store_true")
    p.add_argument("--no-prune", action="store_true")
    p.add_argument("--no-quantize", action="store_true")
    p.add_argument("--no-qat", action="store_true",
                   help="退回纯 PTQ 静态量化（掉点会远超 3%%，仅用于没有训练条件时对比）")
    p.add_argument("--qat-epochs", type=int, default=6, help="量化感知训练轮数")
    p.add_argument("--qat-lr", type=float, default=3e-5, help="量化感知训练骨干学习率")
    p.add_argument("--per-tensor", action="store_true", help="用逐张量量化代替逐通道（更快但精度略降）")
    p.add_argument("--calibration-samples", type=int, default=128)
    p.add_argument("--out", help="产物根目录（默认 HEYAN_HOME 或 ./artifacts）")
    p.add_argument("--region", default="粤东西北")
    p.add_argument("--notes", default="")
    p.add_argument("--pack-zip", action="store_true")
    p.add_argument("--benchmark-rounds", type=int, default=20)
    p.add_argument("--benchmark-threads", type=int, default=1)
    p.add_argument("--voice-langs", default="zh", help="逗号分隔，如 zh,yue")
    p.add_argument("--no-voicepack", action="store_true")
    p.add_argument("--device-ram-mb", type=int)
    p.add_argument("--device-cost-cny", type=float)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("demo-data", help="生成合成演示数据集")
    p.add_argument("--out", default="artifacts/data/demo_dataset")
    p.add_argument("--per-class", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--labels-csv", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_demo_data)

    p = sub.add_parser("ingest", help="导入田间照片（labels.csv）并做体检")
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--out", default="artifacts/data/field")
    p.add_argument("--min-per-class", type=int, default=5)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("recognize", help="单张图离线识别")
    p.add_argument("image")
    p.add_argument("--bundle")
    p.add_argument("--backend", choices=["auto", "onnxruntime", "numpy", "torch"])
    p.add_argument("--lang", default="zh")
    p.add_argument("--brief", action="store_true", help="只打印结论与建议")
    p.set_defaults(func=cmd_recognize)

    p = sub.add_parser("info", help="查看模型包元信息")
    p.add_argument("--bundle")
    p.add_argument("--probe", action="store_true", help="顺带加载引擎打印运行时信息")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("benchmark", help="复测体积/延迟/内存预算")
    p.add_argument("--bundle")
    p.add_argument("--images", nargs="*", help="用于延迟测量的真实图片")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device-ram-mb", type=int)
    p.add_argument("--device-cost-cny", type=float)
    p.add_argument("--save", help="报告输出路径")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("voicepack", help="离线预渲染语音包")
    p.add_argument("--langs", default="zh,yue,hak,teochew,en")
    p.add_argument("--out", default="artifacts/voicepacks/heyan-voicepack")
    p.add_argument("--rate", type=int, default=0, help="语速，0 为引擎默认")
    p.add_argument("--force", action="store_true", help="重渲染已存在的 wav")
    p.add_argument("--into-bundle", action="store_true", help="渲染后拷入最新 bundle")
    p.set_defaults(func=cmd_voicepack)

    p = sub.add_parser("serve", help="启动本地 Web 界面")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--bundle")
    p.add_argument("--lang", default="zh")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--lan", action="store_true",
                   help="监听所有网卡，让同一局域网的其他电脑/手机能打开（等价 --host 0.0.0.0）")
    p.add_argument("--ssl-cert",
                   help="HTTPS 证书路径（与 --ssl-key 成对）；对方浏览器只有走 HTTPS 才放开实时取景和离线缓存")
    p.add_argument("--ssl-key", help="HTTPS 私钥路径")
    p.add_argument("--access-token",
                   help="访问口令；给了之后所有页面和接口都要带口令（也可用环境变量 HEYAN_ACCESS_TOKEN）")
    p.add_argument("--public", action="store_true",
                   help="按公网暴露处理：开启限流与响应头收紧")
    p.add_argument("--tunnel", action="store_true",
                   help="内网穿透模式：只监听回环 + 强制口令鉴权 + 公网加固，隧道自己起（或用 heyan tunnel）")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("tunnel",
                       help="内网穿透：同时起服务和 cloudflared/ngrok/cpolar，打印可直接分享的公网链接")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--client", choices=["cloudflared", "ngrok", "cpolar"],
                   help="指定隧道客户端，默认自动探测（cloudflared 免注册，ngrok/cpolar 要 authtoken）")
    p.add_argument("--access-token", help="访问口令；不给就自动生成并打印出来")
    p.add_argument("--subdomain", help="固定二级域名（要账号套餐支持；cloudflared 不支持）")
    p.add_argument("--region", help="隧道节点区域，如 ngrok 的 ap、cpolar 的 cn、cloudflared 的 us/eu/ap")
    p.add_argument("--bundle")
    p.add_argument("--lang", default="zh")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="等服务与隧道就绪的最长秒数")
    p.set_defaults(func=cmd_tunnel)

    p = sub.add_parser("export", help="导出识别记录")
    p.add_argument("--fmt", default="json", choices=["json", "csv", "both"])
    p.add_argument("--purpose", default="statistics",
                   choices=["statistics", "insurance", "subsidy", "agri_supply"])
    p.add_argument("--channel", default="local", choices=["local", "usb", "bluetooth", "download"])
    p.add_argument("--crop")
    p.add_argument("--class-id")
    p.add_argument("--since")
    p.add_argument("--until")
    p.add_argument("--anonymize", action="store_true", default=None)
    p.add_argument("--usb-target")
    p.add_argument("--db")
    p.add_argument("--out")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("outbox", help="对接发件箱（保险/补贴/农资）")
    p.add_argument("action", choices=["status", "describe", "schemas", "flush", "dispatch"])
    p.add_argument("--names", help="dispatch 时的适配器名，逗号分隔")
    p.add_argument("--target", help="flush 的目标目录（如 U 盘）")
    p.add_argument("--channel")
    p.add_argument("--out", default="artifacts/schemas")
    p.add_argument("--db")
    p.set_defaults(func=cmd_outbox)

    p = sub.add_parser("schema", help="写出全部 JSON Schema")
    p.add_argument("--out", default="artifacts/schemas")
    p.set_defaults(func=cmd_schema)

    p = sub.add_parser("usability", help="运行可用性量表（SUS+TAM）")
    p.add_argument("--lang", default="zh")
    p.add_argument("--out", default="artifacts/usability")
    p.add_argument("--participant", default="")
    p.set_defaults(func=cmd_usability)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\n[heyan] 已取消", file=sys.stderr)
        return 130
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - CLI 兜底，避免裸 traceback 吓到用户
        print(f"[heyan] 出错：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
