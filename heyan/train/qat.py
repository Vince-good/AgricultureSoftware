"""量化感知训练（QAT）。

为什么不能只做 PTQ：实测把 onnxruntime 的静态量化直接套在这套 MobileNetV3-Small
上（尤其经过低秩剪枝之后），验证集 top1 从 0.705 掉到 0.23~0.53。MinMax /
Percentile / Entropy 三种定标法、per-channel、reduce_range、排除 HardSwish 与
SE 门控，全都救不回来。原因是 SE 块与低秩瓶颈产生的中间激活动态范围极窄，
8-bit 均匀网格把一整段特征压成同一个码字，事后定标无法补偿。

激活定标默认走**百分位截断**而不是 MinMax，这是田间真实照片微调时补上的一课：
MobileNetV3 的残差分支上会出现极少数幅值极大的离群激活（实测首层倒残差的范围
宽到 223，而 99% 的值都落在 20 以内）。MinMax 把网格步长撑到 0.87，一整段有效
特征被压成同一个码字，伪量化后 top1 从 0.809 直接掉到 0.342；同一份权重换成
99.5 百分位定标，最宽范围收到 16，零训练就有 0.783。定标方法带来的差距比训练
本身还大，所以 `QUANT_CALIBRATION_PERCENTILE` 是配置项而不是写死的常量。

百分位取值对结果很敏感，而且敏感性会被采样噪声掩盖，所以定标必须扫着选而不是
拍一个：99.5 在三档采样预算下都收敛到 0.783，99.9 在 0.69~0.75 之间乱跳，
99.95 采样越多反而越差（0.112 -> 0.158 掉点）。采样预算同理——每层蓄水池从
65536 提到 262144 才让 99.5 的估计稳定下来，再往上（1048576）已经没有增益。

文档 3.2(2)(4) 承诺的是"量化后精度损失不超过 3%"。能兑现这个合同的只有一条路：
把伪量化算子放进训练回路，先用真实田间校准图定标，再带伪量化微调若干轮，
让权重自己适应 8-bit 网格。导出时把学到的 scale / zero_point 固化成 ONNX 的
QuantizeLinear / DequantizeLinear，再把权重折叠成 int8 存盘。
训练时跑的是什么，设备上跑的就是什么。
"""
from __future__ import annotations

import collections
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import INPUT_SIZE, QUANT_CALIBRATION_PERCENTILE

ACT_QMIN, ACT_QMAX = 0, 255      # 激活：非对称 uint8
WT_QMIN, WT_QMAX = -128, 127     # 权重：对称 int8，per-channel
OPSET = 17
# 百分位定标时每层最多留多少个采样点。54 层 × 262144 × 4B ≈ 57MB，只在定标那
# 几秒存在，freeze() 里立刻释放。这个预算是扫出来的下限：65536 时 99.5 分位的
# 估计还带着 0.007 的抖动，262144 与 1048576 结果完全一致。
RESERVOIR_SAMPLES = 262144
PER_BATCH_SAMPLES = 32768


class ActivationQuant(nn.Module):
    """per-tensor 非对称 uint8 伪量化 + 可切换的激活观测器。

    定标阶段只观测不量化，冻结后只量化不观测，这样训练用的 scale 与
    写进 ONNX 的 scale 完全一致。

    `percentile` 为空时是原来的滑动平均 MinMax；给了百分位（例如 99.9）就在
    定标期蓄水池采样，冻结时取 [(100-p), p] 分位作为范围上下界（与 onnxruntime
    的 PercentileCalibrator 同一套约定：p=99.9 表示上界切在 99.9 分位、下界切在
    0.1 分位），把离群激活切在网格之外——它们本来就是极少数，切掉的损失远小于
    把整张网格撑粗的损失。
    """

    def __init__(self, name: str = "", averaging: float = 0.05,
                 percentile: Optional[float] = None,
                 reservoir: int = RESERVOIR_SAMPLES) -> None:
        super().__init__()
        self.name = name
        self.averaging = averaging
        self.percentile = percentile
        self.reservoir = int(reservoir)
        self.observer_enabled = True
        self.fake_quant_enabled = False
        self._pool: List[torch.Tensor] = []   # 只在定标期存在，不进 state_dict
        self.register_buffer("scale", torch.ones(1, dtype=torch.float32))
        self.register_buffer("zero_point", torch.zeros(1, dtype=torch.int32))
        self.register_buffer("min_val", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("max_val", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("seen", torch.zeros(1, dtype=torch.float32))

    def reset(self) -> None:
        """重新定标前清空蓄水样本，否则上一轮的分布会混进来。"""
        self._pool = []
        self.seen.zero_()

    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> None:
        if self.percentile is not None:
            self._collect(x.detach())
            return
        bmin = float(x.min().detach())
        bmax = float(x.max().detach())
        if float(self.seen.item()) == 0.0:
            self.min_val.fill_(bmin)
            self.max_val.fill_(bmax)
            self.seen.fill_(1.0)
        else:
            a = self.averaging
            self.min_val.mul_(1.0 - a).add_(a * bmin)
            self.max_val.mul_(1.0 - a).add_(a * bmax)

    @torch.no_grad()
    def _collect(self, x: torch.Tensor) -> None:
        """蓄水池采样：每张图每层只留 PER_BATCH_SAMPLES 个点，够估分位数就行。

        必须随机取下标，不能等距切片：激活张量按 NCHW 摊平，等距步长很容易正好
        等于 H*W 的约数（实测首层倒残差 3211264//8192=392，而 112*112/392=32），
        于是每张图每个通道都只采到同样那 32 个像素位置，分位数估计会带系统性偏差。
        """
        if sum(int(t.numel()) for t in self._pool) >= self.reservoir:
            return
        flat = x.reshape(-1)
        if flat.numel() > PER_BATCH_SAMPLES:
            idx = torch.randint(flat.numel(), (PER_BATCH_SAMPLES,))
            flat = flat[idx]
        self._pool.append(flat.to(dtype=torch.float32, copy=True))

    def freeze(self) -> None:
        """把观测到的范围换算成 uint8 仿射参数，零点必须落在 [0,255]。"""
        if self.percentile is not None:
            self._apply_percentile()
        mn = min(float(self.min_val.item()), 0.0)
        mx = max(float(self.max_val.item()), 0.0)
        scale = max(mx - mn, 1e-8) / float(ACT_QMAX - ACT_QMIN)
        zp = int(round(-mn / scale))
        self.scale.fill_(scale)
        self.zero_point.fill_(max(ACT_QMIN, min(ACT_QMAX, zp)))

    @torch.no_grad()
    def _apply_percentile(self) -> None:
        if not self._pool:
            return                      # 没定标过就沿用现有 min/max，不猜
        cat = torch.cat([t.reshape(-1) for t in self._pool])
        tail = (100.0 - float(self.percentile)) / 100.0   # 99.9 -> 上下各切 0.1%
        lo = float(torch.quantile(cat, tail))
        hi = float(torch.quantile(cat, 1.0 - tail))
        if hi <= lo:                    # 整层几乎是常量，退回真实极值免得网格退化
            lo, hi = float(cat.min()), float(cat.max())
        self.min_val.fill_(lo)
        self.max_val.fill_(hi)
        self.seen.fill_(1.0)
        self._pool = []                 # 定标结束立刻放掉这十几 MB

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.observer_enabled:
            self.observe(x)
        if self.fake_quant_enabled:
            return torch.fake_quantize_per_tensor_affine(
                x, float(self.scale.item()), int(self.zero_point.item()), ACT_QMIN, ACT_QMAX)
        return x


class QuantizedLayer(nn.Module):
    """把 Conv2d / Linear 包一层：输入激活与权重都走伪量化。

    权重 scale 在训练期按当前权重实时重算（对称 per-channel），导出前冻结成
    常量，这样 ONNX 图里的 Q/DQ 是可折叠成 int8 的静态参数。
    """

    def __init__(self, inner: nn.Module, name: str = "", quant_input: bool = True,
                 percentile: Optional[float] = QUANT_CALIBRATION_PERCENTILE) -> None:
        super().__init__()
        if not isinstance(inner, (nn.Conv2d, nn.Linear)):
            raise TypeError(f"QuantizedLayer 只能包 Conv2d/Linear，收到 {type(inner)}")
        self.inner = inner
        self.name = name
        self.weight_dynamic = True
        self.bypass = False          # True 时整层退化成普通浮点算子（重导 FP32 参照物用）
        self.percentile = percentile
        self.act_in = (ActivationQuant(f"{name}.in", percentile=self.percentile)
                       if quant_input else None)
        self.register_buffer("w_scale", torch.ones(inner.weight.shape[0], dtype=torch.float32))
        self.register_buffer("w_zp", torch.zeros(inner.weight.shape[0], dtype=torch.int32))
        self._update_weight_scale(inner.weight.detach())

    @torch.no_grad()
    def _update_weight_scale(self, w: torch.Tensor) -> None:
        flat = w.reshape(w.shape[0], -1)
        amax = flat.abs().amax(dim=1).clamp_min(1e-8)
        self.w_scale.copy_(amax / float(WT_QMAX))
        self.w_zp.zero_()

    def freeze_weights(self) -> None:
        self._update_weight_scale(self.inner.weight.detach())
        self.weight_dynamic = False

    def quantized_weight(self) -> torch.Tensor:
        w = self.inner.weight
        if self.bypass:
            return w
        if self.weight_dynamic:
            self._update_weight_scale(w.detach())
        return torch.fake_quantize_per_channel_affine(
            w, self.w_scale, self.w_zp, 0, WT_QMIN, WT_QMAX)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_in is not None:
            x = self.act_in(x)
        w = self.quantized_weight()
        c = self.inner
        if isinstance(c, nn.Conv2d):
            return F.conv2d(x, w, c.bias, c.stride, c.padding, c.dilation, c.groups)
        return F.linear(x, w, c.bias)


def iter_quantizers(model: nn.Module) -> Iterable[ActivationQuant]:
    for m in model.modules():
        if isinstance(m, ActivationQuant):
            yield m


def set_mode(model: nn.Module, observer: bool, fake_quant: bool) -> int:
    n = 0
    for m in iter_quantizers(model):
        m.observer_enabled = observer
        m.fake_quant_enabled = fake_quant
        if observer:
            m.reset()      # 重新进入定标态就把上一轮的蓄水样本清掉
        n += 1
    for m in model.modules():
        if isinstance(m, QuantizedLayer):
            m.weight_dynamic = True
            m.bypass = not fake_quant
    return n


def freeze_scales(model: nn.Module) -> None:
    for m in iter_quantizers(model):
        m.freeze()


def prepare(model: nn.Module, quant_input: bool = True,
            percentile: Optional[float] = QUANT_CALIBRATION_PERCENTILE) -> int:
    """就地把所有 Conv2d / Linear 换成 QuantizedLayer，返回替换数量。"""
    count = 0

    def walk(mod: nn.Module, prefix: str) -> None:
        nonlocal count
        for name, child in list(mod.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                setattr(mod, name, QuantizedLayer(child, full, quant_input,
                                                  percentile=percentile))
                count += 1
            else:
                walk(child, full)

    walk(model, "")
    return count


@torch.no_grad()
def calibrate(model: nn.Module, arrays: Sequence[np.ndarray], batch_size: int = 16,
              percentile: Optional[float] = None) -> int:
    """用真实田间图片定标激活范围。只观测、不量化、不反传。

    `percentile` 只在需要临时改定标方式时传；正常路径由 `prepare()` 决定，
    留空表示沿用每层观测器自己的设置。
    """
    model.eval()
    if percentile is not None:
        for m in iter_quantizers(model):
            m.percentile = percentile
    set_mode(model, observer=True, fake_quant=False)
    used = 0
    for i in range(0, len(arrays), batch_size):
        chunk = np.concatenate(list(arrays[i:i + batch_size]), axis=0)
        model(torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32)))
        used += int(chunk.shape[0])
    freeze_scales(model)
    return used


def quantizer_table(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """把定标结果单独存一份，便于排查"哪个激活范围最窄"。"""
    out: Dict[str, Dict[str, Any]] = {}
    for i, m in enumerate(iter_quantizers(model)):
        out[m.name or f"#{i}"] = {
            "scale": round(float(m.scale.item()), 8),
            "zero_point": int(m.zero_point.item()),
            "min": round(float(m.min_val.item()), 6),
            "max": round(float(m.max_val.item()), 6),
            "observer": (f"percentile_{m.percentile}" if m.percentile is not None
                         else "minmax_ema"),
        }
    return out


# --------------------------------------------------------------------------
# ONNX 后处理：常量折叠 + 权重转 int8 存储
# --------------------------------------------------------------------------

_NP_DTYPE = {1: np.float32, 2: np.uint8, 3: np.int8, 5: np.int16, 6: np.int32,
             7: np.int64, 9: np.bool_, 10: np.float16, 11: np.float64,
             12: np.uint32, 13: np.uint64}


def _fold_static_nodes(model) -> int:
    """把 Constant / Cast 这类纯常量子图折成 initializer。

    torch 导出的伪量化 scale 会带上 Cast(Constant) 链，节点数翻倍且不便检查。
    折成常量后图里只剩 QuantizeLinear / DequantizeLinear 本体。
    """
    import onnx
    from onnx import numpy_helper

    graph = model.graph
    consts: Dict[str, np.ndarray] = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    for n in graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value" and a.type == onnx.AttributeProto.TENSOR:
                    consts[n.output[0]] = numpy_helper.to_array(a.t)

    removable: List[Any] = []
    for n in list(graph.node):
        if n.op_type == "Identity" and n.input and n.input[0] in consts:
            consts[n.output[0]] = consts[n.input[0]]
            removable.append(n)
        elif n.op_type == "Cast" and n.input and n.input[0] in consts:
            to = int(next((a.i for a in n.attribute if a.name == "to"), 0))
            np_dt = _NP_DTYPE.get(to)
            if np_dt is None:
                continue
            consts[n.output[0]] = consts[n.input[0]].astype(np_dt)
            removable.append(n)

    for n in removable:
        graph.node.remove(n)

    # 折掉 Cast/Identity 之后，原来的 Constant 节点就没人用了，一并清掉
    still_used = {inp for node in graph.node for inp in node.input if inp}
    for n in [x for x in graph.node
              if x.op_type == "Constant" and not (set(x.output) & still_used)]:
        graph.node.remove(n)
        removable.append(n)

    used = {inp for node in graph.node for inp in node.input if inp}
    existing = {i.name for i in graph.initializer}
    added = 0
    for name, arr in consts.items():
        if name in used and name not in existing:
            graph.initializer.append(numpy_helper.from_array(arr, name=name))
            existing.add(name)
            added += 1

    produced_by_const = {n.output[0] for n in removable if n.op_type == "Constant"}
    if produced_by_const:
        keep = [i for i in graph.initializer if i.name not in produced_by_const]
        del graph.initializer[:]
        graph.initializer.extend(keep)
    return len(removable)


def fold_qdq_weights(onnx_path: Path | str) -> Dict[str, Any]:
    """把 `float权重 -> QuantizeLinear -> DequantizeLinear` 折叠成 int8 常量。

    torch 导出的 QDQ 图里权重仍是 float32 initializer，文件体积和 FP32 一样大。
    权重侧的 Q/DQ 是完全静态的，可以直接算出量化结果原地替换，体积立刻降到
    约 1/4，onnxruntime 加载时也省掉一次转换。
    """
    import onnx
    from onnx import numpy_helper

    onnx_path = Path(onnx_path)
    model = onnx.load(str(onnx_path))
    folded_consts = _fold_static_nodes(model)
    graph = model.graph
    consts: Dict[str, np.ndarray] = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    producer = {out: n for n in graph.node for out in n.output}
    consumers = collections.Counter(inp for n in graph.node for inp in n.input if inp)

    new_inits: List[Any] = []
    dead_q: List[Any] = []
    freed: List[str] = []
    int8_bytes = 0

    for node in list(graph.node):
        if node.op_type != "DequantizeLinear":
            continue
        q = producer.get(node.input[0])
        if q is None or q.op_type != "QuantizeLinear":
            continue
        w = consts.get(q.input[0])
        scale = consts.get(q.input[1]) if len(q.input) > 1 and q.input[1] else None
        zp = consts.get(q.input[2]) if len(q.input) > 2 and q.input[2] else None
        if w is None or scale is None or w.dtype != np.float32:
            continue

        axis = int(next((a.i for a in q.attribute if a.name == "axis"), 1))
        if not 0 <= axis < max(w.ndim, 1):
            axis = 0
        n_chan = w.shape[axis] if w.ndim > 0 else 1
        scale = np.asarray(scale, dtype=np.float64).ravel()
        if scale.size == 1:
            shape = [1] * max(w.ndim, 1)
            scale_b = scale.reshape(shape)
            z_b = np.zeros_like(scale_b)
            axis = 0
        elif scale.size == n_chan:
            shape = [1] * w.ndim
            shape[axis] = n_chan
            scale_b = scale.reshape(shape)
            z_b = (np.zeros(scale.size, dtype=np.float64) if zp is None
                   else np.asarray(zp, dtype=np.float64).ravel().reshape(shape))
        else:
            continue

        signed = zp is not None and np.asarray(zp).dtype.kind in "iu" and \
            np.asarray(zp).dtype != np.uint8
        qmin, qmax = (WT_QMIN, WT_QMAX) if signed else (ACT_QMIN, ACT_QMAX)
        out_np = np.clip(np.rint(w.astype(np.float64) / scale_b) + z_b, qmin, qmax)
        out_np = out_np.astype(np.int8 if signed else np.uint8)

        qname = f"{q.input[0]}__qfold"
        new_inits.append(numpy_helper.from_array(out_np, name=qname))
        node.input[0] = qname
        if not any(a.name == "axis" for a in node.attribute):
            node.attribute.append(onnx.helper.make_attribute("axis", int(axis)))
        dead_q.append(q)
        int8_bytes += int(out_np.nbytes)
        if consumers[q.input[0]] <= 1:
            freed.append(q.input[0])

    for q in dead_q:
        graph.node.remove(q)
    if new_inits:
        graph.initializer.extend(new_inits)
    if freed:
        drop = set(freed)
        keep = [i for i in graph.initializer if i.name not in drop]
        del graph.initializer[:]
        graph.initializer.extend(keep)

    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, str(onnx_path))
    return {"folded_layers": len(dead_q), "int8_bytes": int8_bytes,
            "freed_float_tensors": len(freed), "constant_nodes_folded": folded_consts,
            "size_mb": round(onnx_path.stat().st_size / (1024 * 1024), 3)}


def export_qat_onnx(model: nn.Module, out_path: Path | str,
                    input_size: int = INPUT_SIZE, opset: int = OPSET) -> Dict[str, Any]:
    """导出带 QDQ 的 ONNX，并把权重折叠成 int8。"""
    import onnx

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    set_mode(model, observer=False, fake_quant=True)
    for m in model.modules():
        if isinstance(m, QuantizedLayer):
            m.freeze_weights()

    dummy = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
    kwargs: Dict[str, Any] = dict(
        input_names=["input"], output_names=["logits"], export_params=True,
        opset_version=opset, do_constant_folding=False,
    )
    try:
        torch.onnx.export(model, (dummy,), str(out_path), dynamo=False, **kwargs)
    except TypeError:  # pragma: no cover - 老版本 torch 没有 dynamo 参数
        torch.onnx.export(model, (dummy,), str(out_path), **kwargs)

    fold_info = fold_qdq_weights(out_path)
    loaded = onnx.load(str(out_path))
    ops = collections.Counter(n.op_type for n in loaded.graph.node)
    info = {
        "path": str(out_path), "input_size": input_size, "opset": opset,
        "size_mb": round(out_path.stat().st_size / (1024 * 1024), 3),
        "nodes": len(loaded.graph.node), "ops": dict(ops), "weight_folding": fold_info,
    }
    print(f"[qat] QDQ ONNX -> {info['path']} ({info['size_mb']} MB, {info['nodes']} 节点, "
          f"折叠 {fold_info['folded_layers']} 层权重为 int8)")
    return info


# --------------------------------------------------------------------------
# QAT 训练
# --------------------------------------------------------------------------

def run_qat(model: nn.Module, train_ds, val_ds, class_ids: Sequence[str], cfg,
            calib_arrays: Sequence[np.ndarray], work: Path | str,
            epochs: int = 6, batch_size: int = 16, lr: float = 3e-5,
            verbose: bool = True) -> Dict[str, Any]:
    """定标 -> 伪量化微调。复用 finetune.train 的训练回路（早停/余弦退火/类别均衡）。

    返回 dict：`model` 是带伪量化的 torch 模块，可直接交给 export_qat_onnx。
    """
    from . import finetune

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    replaced = prepare(model)
    if replaced == 0:
        raise RuntimeError("模型里没有 Conv2d/Linear，无法做 QAT")
    n_calib = calibrate(model, calib_arrays, batch_size=batch_size)
    set_mode(model, observer=False, fake_quant=True)
    if verbose:
        print(f"[qat] 插入伪量化层 {replaced} 个，用 {n_calib} 张校准图定标完成")

    overrides = dict(cfg.to_dict())
    overrides.update({
        "epochs": int(epochs), "batch_size": int(batch_size), "strategy": "qat",
        "head_lr": lr * 30.0, "backbone_lr": lr, "label_smoothing": 0.05,
        "mixup_alpha": 0.0, "patience": max(2, int(epochs)),
        "out_dir": str(work), "tag": "qat", "num_classes": len(class_ids),
    })
    qcfg = cfg.__class__(**overrides)

    res = finetune.train(train_ds, val_ds, list(class_ids), qcfg,
                         init_model=model, verbose=verbose)
    qmodel = res.model if res.model is not None else model
    set_mode(qmodel, observer=False, fake_quant=True)
    for m in qmodel.modules():
        if isinstance(m, QuantizedLayer):
            m.freeze_weights()

    payload = {
        "layers_quantized": replaced,
        "calibration_samples": int(n_calib),
        "epochs": int(epochs),
        "best_epoch": res.best_epoch,
        "elapsed_s": round(time.time() - t0, 1),
        "metrics": {k: v for k, v in res.metrics.items() if k != "confusion_matrix"},
        "activations": quantizer_table(qmodel),
        "history": res.history,
    }
    (work / "qat_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return {"model": qmodel, "report": payload, "checkpoint": res.checkpoint}


def qat_quantize_and_verify(model: nn.Module, fp32_path: Path | str, int8_path: Path | str,
                            calib_arrays: Sequence[np.ndarray], val_arrays: Sequence[np.ndarray],
                            val_labels: Sequence[int], class_ids: Sequence[str],
                            train_ds, val_ds, base_cfg, work: Path | str,
                            epochs: int = 6, batch_size: int = 16, lr: float = 3e-5,
                            threads: int = 1, reexport_fp32: bool = True,
                            ptq_fallback: bool = True, verbose: bool = True):
    """QAT 量化 + 三重验收（体积/精度/延迟），返回 quantize.QuantizeReport。

    训练完的网络权重已经适配过 8-bit 网格，所以 FP32 参照物也从同一份权重导出
    （关掉伪量化再导一次）。这样 bundle 里的 FP32 与 INT8 是同一个网络，
    "精度损失"度量的才是纯量化误差，而不是混进了训练差异。
    """
    import copy

    from . import quantize
    from .export_onnx import export_module_onnx
    from ..config import (ACCURACY_DROP_MAX, LATENCY_MAX_S, MODEL_SIZE_MAX_MB)

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    fp32_path = Path(fp32_path)
    int8_path = Path(int8_path)
    int8_path.parent.mkdir(parents=True, exist_ok=True)

    qmodel = copy.deepcopy(model)
    out = run_qat(qmodel, train_ds, val_ds, class_ids, base_cfg, calib_arrays, work,
                  epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)
    qmodel = out["model"]

    export_info = export_qat_onnx(qmodel, int8_path, input_size=INPUT_SIZE)

    if reexport_fp32:
        # bypass=True 时 QuantizedLayer 退化成普通浮点算子，导出的就是同一份权重的 FP32 图
        set_mode(qmodel, observer=False, fake_quant=False)
        export_module_onnx(qmodel, fp32_path, input_size=INPUT_SIZE)
        set_mode(qmodel, observer=False, fake_quant=True)

    report = quantize.QuantizeReport(
        method="qat_qdq_int8",
        fp32_path=str(fp32_path), int8_path=str(int8_path),
        fp32_size_mb=round(fp32_path.stat().st_size / (1024 * 1024), 3),
        calibration_samples=len(calib_arrays), per_channel=True,
    )
    report.int8_size_mb = export_info["size_mb"]
    report.compression_ratio = round(report.fp32_size_mb / max(report.int8_size_mb, 1e-6), 2)
    report.notes.append(f"QAT {epochs} 轮，伪量化层 {out['report']['layers_quantized']} 个，"
                        f"权重折叠 {export_info['weight_folding']['folded_layers']} 层为 int8")

    if val_arrays:
        labels = np.asarray(val_labels, dtype=np.int64)
        logits32, t32 = quantize.run_onnx(fp32_path, val_arrays, threads=threads)
        logits8, t8 = quantize.run_onnx(int8_path, val_arrays, threads=threads)
        report.metrics_fp32 = {k: v for k, v in
                               quantize.summarize(logits32, labels, class_ids).items()
                               if k != "confusion_matrix"}
        report.metrics_int8 = {k: v for k, v in
                               quantize.summarize(logits8, labels, class_ids).items()
                               if k != "confusion_matrix"}
        report.accuracy_drop_top1 = round(
            report.metrics_fp32["top1"] - report.metrics_int8["top1"], 4)
        report.latency_fp32_ms = quantize.latency_stats(t32)["p50"]
        report.latency_int8_ms = quantize.latency_stats(t8)["p50"]
        report.speedup = round(report.latency_fp32_ms / max(report.latency_int8_ms, 1e-6), 2)
        # torch 侧与 onnxruntime 侧必须一致，否则说明导出图与训练图不等价
        with torch.no_grad():
            set_mode(qmodel, observer=False, fake_quant=True)
            for m in qmodel.modules():
                if isinstance(m, QuantizedLayer):
                    m.freeze_weights()
            torch_logits = np.concatenate(
                [qmodel(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
                        ).detach().numpy() for a in val_arrays], axis=0)
        report.metrics_int8["top1_torch"] = quantize.summarize(
            torch_logits, labels, class_ids)["top1"]
        report.notes.append(
            f"torch/ORT 一致性: top1 {report.metrics_int8['top1_torch']:.4f} vs "
            f"{report.metrics_int8['top1']:.4f}")

    if ptq_fallback and report.accuracy_drop_top1 > ACCURACY_DROP_MAX and val_arrays:
        # QAT 仍不达标时再试一次 PTQ，取精度更好的那个，不掩盖问题
        cand = work / "model_int8_ptq.onnx"
        try:
            ptq = quantize.quantize_and_verify(fp32_path, cand, calib_arrays, val_arrays,
                                               val_labels, class_ids, threads=threads,
                                               max_model_mb=MODEL_SIZE_MAX_MB,
                                               max_latency_s=LATENCY_MAX_S)
            if ptq.accuracy_drop_top1 < report.accuracy_drop_top1:
                print(f"[qat] PTQ 掉点更少（{ptq.accuracy_drop_top1} < "
                      f"{report.accuracy_drop_top1}），改用 PTQ 产物")
                import shutil
                shutil.copyfile(cand, int8_path)
                report.method = ptq.method
                report.int8_size_mb = ptq.int8_size_mb
                report.metrics_int8 = ptq.metrics_int8
                report.accuracy_drop_top1 = ptq.accuracy_drop_top1
                report.latency_int8_ms = ptq.latency_int8_ms
                report.notes.append("已回退到 PTQ 产物")
        except Exception as exc:  # pragma: no cover
            print(f"[qat] PTQ 兜底失败，保留 QAT 产物: {type(exc).__name__}: {exc}")

    checks = {
        "model_size": {"value_mb": report.int8_size_mb, "limit_mb": MODEL_SIZE_MAX_MB,
                       "ok": report.int8_size_mb <= MODEL_SIZE_MAX_MB},
        "accuracy_drop": {"value": report.accuracy_drop_top1, "limit": ACCURACY_DROP_MAX,
                          "ok": report.accuracy_drop_top1 <= ACCURACY_DROP_MAX},
        "latency": {"value_ms": report.latency_int8_ms, "limit_ms": LATENCY_MAX_S * 1000.0,
                    "ok": report.latency_int8_ms <= LATENCY_MAX_S * 1000.0},
    }
    report.budgets = checks
    report.ok = all(c["ok"] for c in checks.values())
    for name, c in checks.items():
        if not c["ok"]:
            report.notes.append(f"预算超标: {name} -> {c}")

    print(f"[qat] 体积 {report.fp32_size_mb}MB -> {report.int8_size_mb}MB "
          f"(x{report.compression_ratio}), top1 {report.metrics_fp32.get('top1', '-')} -> "
          f"{report.metrics_int8.get('top1', '-')} (掉 {report.accuracy_drop_top1}), "
          f"延迟 p50 {report.latency_fp32_ms}ms -> {report.latency_int8_ms}ms, "
          f"验收{'通过' if report.ok else '未通过'}")
    return report
