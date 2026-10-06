"""The networks nn_bench runs, named by their shape, and the groups bench.yaml uses.

A name is fields joined by '-', the input first:

    mlp-12to256to256to4                            input length, hidden widths, outputs
    mobilenetv1-160x120c1-a025-c8to256-out1000     MobileNet v1: a W x H x C input, width multiplier
                                                   alpha (a025: 0.25), channels from the stem to
                                                   the last pointwise layer, outputs
    resnet18-320x240c1-c16to128-out1000            ResNet-18 (basic blocks): stage 0 at 16 channels,
                                                   doubling to 128 in stage 3
    convpool-160x96c1-c8s2mp-c16mp-c32mp-c32-out4  3x3 convolution units
    dwpw-24x24c16-c32mp-c64mp-c96mp-out4           depthwise-separable units
    resblock-16x16c3-c8-c8-c16mp-c16-c16-out4      residual units after an implied 3x3 stem

A unit c<C>[s2][mp] outputs C channels; s2: its first 3x3 has stride 2; mp: then a
2x2 max pool. Any mlp, mobilenetv1 or resnet18 name of these forms is a network;
the three unit chains are fixed (FIXED), the first two also the fused executors'.
GROUPS only collects the names the sets use. Weights are random (graph.py): what
is measured is how a network of this shape runs, and whether it computes exactly
what it specifies.
"""
from __future__ import annotations

import re

from graph import Graph


def mlp(dims: list[int]) -> Graph:
    g = Graph((dims[0],))
    for i, n in enumerate(dims[1:-1]):
        g.fc(n, label=f"fc{i}")
    g.fc(dims[-1], requant=False, label="out")
    return g


def mobilenet_v1(alpha: float, height: int, width: int, channels: int, classes: int) -> Graph:
    """MobileNet v1: a stride-2 stem, then 13 depthwise-separable blocks."""
    ch = lambda c: max(8, int(c * alpha))  # noqa: E731
    g = Graph((channels, height, width))
    g.conv(ch(32), 3, 2, "stem")
    blocks = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2)] + [(512, 1)] * 5 \
        + [(1024, 2), (1024, 1)]
    for i, (c, stride) in enumerate(blocks):
        g.depthwise(stride, f"block{i}_dw")
        g.conv(ch(c), 1, 1, f"block{i}_pw")
    g.global_average()
    g.fc(classes, requant=False, label="out")
    return g


def resnet18(base: int, height: int, width: int, channels: int, classes: int) -> Graph:
    """ResNet-18: a stride-2 stem and a pool, then four stages of two basic
    blocks, each stage twice as wide and half the size; a 1x1 convolution
    projects the skip where the shape changes."""
    g = Graph((channels, height, width))
    g.conv(base, 3, 2, "stem")
    g.maxpool2("stem_pool")
    for stage in range(4):
        c = base << stage
        for block in range(2):
            stride = 2 if stage and not block else 1
            name = f"stage{stage}_block{block}"
            skip = g.branch
            if stride == 2:
                g.conv(c, 1, 2, f"{name}_proj")
                skip, (g.tensor, g.values) = g.branch, skip
            g.conv(c, 3, stride, f"{name}_conv0")
            g.conv(c, 3, 1, f"{name}_conv1")
            g.add(skip, f"{name}_add")
    g.global_average()
    g.fc(classes, requant=False, label="out")
    return g


def convpool() -> Graph:
    """convpool-160x96c1-c8s2mp-c16mp-c32mp-c32-out4."""
    g = Graph((1, 96, 160))
    units = ((8, 2, True), (16, 1, True), (32, 1, True), (32, 1, False))  # channels, stride, pooled
    for i, (c, stride, pooled) in enumerate(units):
        g.conv(c, 3, stride, f"unit{i}_conv")
        if pooled:
            g.maxpool2(f"unit{i}_pool")
    g.global_average()
    g.fc(4, requant=False, label="out")
    return g


def dwpw() -> Graph:
    """dwpw-24x24c16-c32mp-c64mp-c96mp-out4."""
    g = Graph((16, 24, 24))
    for i, c in enumerate((32, 64, 96)):
        g.depthwise(1, f"unit{i}_dw")
        g.conv(c, 1, 1, f"unit{i}_pw")
        g.maxpool2(f"unit{i}_pool")
    g.global_average()
    g.fc(4, requant=False, label="out")
    return g


def resblock() -> Graph:
    """resblock-16x16c3-c8-c8-c16mp-c16-c16-out4."""
    g = Graph((3, 16, 16))
    g.conv(8, 3, 1, "stem")
    units = ((8, False), (8, False), (16, True), (16, False), (16, False))  # channels, pooled
    for i, (c, pooled) in enumerate(units):
        skip = g.branch
        if skip[1].shape[0] != c:
            g.conv(c, 1, 1, f"unit{i}_proj")
            skip, (g.tensor, g.values) = g.branch, skip
        g.conv(c, 3, 1, f"unit{i}_conv0")
        g.conv(c, 3, 1, f"unit{i}_conv1")
        g.add(skip, f"unit{i}_add")
        if pooled:
            g.maxpool2(f"unit{i}_pool")
    g.global_average()
    g.fc(4, requant=False, label="out")
    return g


FIXED = {"convpool-160x96c1-c8s2mp-c16mp-c32mp-c32-out4": convpool,
         "dwpw-24x24c16-c32mp-c64mp-c96mp-out4": dwpw,
         "resblock-16x16c3-c8-c8-c16mp-c16-c16-out4": resblock}
INPUT = r"(\d+)x(\d+)c(\d+)"  # width x height, channels


def build(name: str) -> Graph:
    if name in FIXED:
        return FIXED[name]()
    if m := re.fullmatch(r"mlp-(\d+(?:to\d+)+)", name):
        return mlp([int(d) for d in m.group(1).split("to")])
    if m := re.fullmatch(rf"mobilenetv1-{INPUT}-a(\d{{3}})-c\d+to\d+-out(\d+)", name):
        width, height, channels, alpha, classes = (int(v) for v in m.groups())
        return mobilenet_v1(alpha / 100, height, width, channels, classes)
    if m := re.fullmatch(rf"resnet18-{INPUT}-c(\d+)to\d+-out(\d+)", name):
        width, height, channels, base, classes = (int(v) for v in m.groups())
        return resnet18(base, height, width, channels, classes)
    raise ValueError(f"no network named {name!r} (see networks.py)")


# A camera network is listed without its input field (resnet18-c8to64-out1000), and
# a bench.yaml entry's `inputs` says which inputs it runs on.
INPUTS = ("160x120c1", "320x240c1")
TEMPLATE = r"(mobilenetv1)-(a\d{3}-c\d+to\d+-out\d+)|(resnet18)-(c\d+to\d+-out\d+)"
GROUPS = {
    # Widths across DORY's L1 tiling threshold (~192), square beside non-square.
    "mlp_fits_l2": ["mlp-12to64to64to4", "mlp-12to128to128to4", "mlp-12to192to192to4",
                    "mlp-12to256to256to4", "mlp-12to384to384to4", "mlp-12to256to128to4",
                    "mlp-12to128to256to4", "mlp-12to256to4", "mlp-12to256to256to256to4"],
    # Weights past L2: only l3 builds them; the last two stream weight tiles
    # from HyperRAM inside a layer.
    "mlp_needs_l3": ["mlp-12to512to512to4", "mlp-12to512to512to512to4", "mlp-12to768to768to4",
                     "mlp-12to1024to1024to4"],
    "custom_cnns": ["resblock-16x16c3-c8-c8-c16mp-c16-c16-out4",
                    "dwpw-24x24c16-c32mp-c64mp-c96mp-out4",
                    "convpool-160x96c1-c8s2mp-c16mp-c32mp-c32-out4"],
    # Up to the widest DORY runs on both inputs (README, Networks). resnet18 goes
    # to c28to224 on 160x120c1 alone; bench.yaml lists that one by name.
    "mobilenetv1_widths": ["mobilenetv1-a025-c8to256-out1000", "mobilenetv1-a050-c16to512-out1000",
                           "mobilenetv1-a075-c24to768-out1000",
                           "mobilenetv1-a100-c32to1024-out1000"],
    "resnet18_widths": ["resnet18-c8to64-out1000", "resnet18-c16to128-out1000",
                        "resnet18-c20to160-out1000"],
}


def names(spec, inputs=INPUTS) -> list[str]:
    """Network names from a group, a name, or a list of either; a camera network
    once per input, the input inserted after the family."""
    out = []
    for item in [spec] if isinstance(spec, str) else spec:
        for name in GROUPS.get(item, [item]):
            if m := re.fullmatch(TEMPLATE, name):
                family, rest = [group for group in m.groups() if group]
                out += [f"{family}-{i}-{rest}" for i in inputs]
            else:
                out.append(name)
    return out
