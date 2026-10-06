#!/usr/bin/env python3
"""Generate model-specific raw-byte activation or softmax-exp LUTs."""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _hard_sigmoid(value: float) -> float:
    return min(1.0, max(0.0, value / 6.0 + 0.5))


FUNCTIONS: dict[str, Callable[[float], float]] = {
    "sigmoid": _sigmoid,
    "tanh": math.tanh,
    "hard_sigmoid": _hard_sigmoid,
    "hard_swish": lambda value: value * _hard_sigmoid(value),
    "silu": lambda value: value * _sigmoid(value),
    "gelu": lambda value: value
    * 0.5
    * (1.0 + math.erf(value / math.sqrt(2.0))),
}


def _round_away(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5)


def activation_lut(
    mode: str,
    input_scale: float,
    input_zero_point: int,
    input_signed: bool,
    output_scale: float,
    output_zero_point: int,
    output_signed: bool,
    leaky_slope: float = 1.0 / 8.0,
) -> list[int]:
    if mode == "leaky_relu":
        def function(value: float) -> float:
            return value if value >= 0.0 else value * leaky_slope

    else:
        function = FUNCTIONS[mode]
    lower, upper = (-128, 127) if output_signed else (0, 255)
    result: list[int] = []
    for raw in range(256):
        quantized_input = raw - 256 if input_signed and raw >= 128 else raw
        real_input = (quantized_input - input_zero_point) * input_scale
        quantized_output = (
            _round_away(function(real_input) / output_scale) + output_zero_point
        )
        quantized_output = min(upper, max(lower, quantized_output))
        result.append(quantized_output & 0xFF)
    return result


def softmax_lut(input_scale: float) -> list[int]:
    return [
        min(32768, _round_away(math.exp(-difference * input_scale) * 32768.0))
        for difference in range(256)
    ]


def emit(symbol: str, values: list[int], value_type: str) -> None:
    width = 4 if value_type == "uint16_t" else 2
    print(f"const {value_type} {symbol}[256] = {{")
    for start in range(0, len(values), 16):
        formatted = ", ".join(
            f"0x{value:0{width}x}u" for value in values[start : start + 16]
        )
        print(f"    {formatted},")
    print("};")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=[*FUNCTIONS, "leaky_relu", "softmax_exp"], required=True
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--input-scale", type=float, required=True)
    parser.add_argument("--input-zero-point", type=int, default=0)
    parser.add_argument("--input-signed", action="store_true")
    parser.add_argument("--output-scale", type=float, default=1.0 / 16.0)
    parser.add_argument("--output-zero-point", type=int, default=0)
    parser.add_argument("--output-signed", action="store_true")
    parser.add_argument("--leaky-slope", type=float, default=1.0 / 8.0)
    args = parser.parse_args()

    if args.input_scale <= 0.0 or args.output_scale <= 0.0:
        parser.error("input and output scales must be positive")
    if args.leaky_slope < 0.0:
        parser.error("leaky slope must be non-negative")

    if args.mode == "softmax_exp":
        emit(args.symbol, softmax_lut(args.input_scale), "uint16_t")
    else:
        emit(
            args.symbol,
            activation_lut(
                args.mode,
                args.input_scale,
                args.input_zero_point,
                args.input_signed,
                args.output_scale,
                args.output_zero_point,
                args.output_signed,
                args.leaky_slope,
            ),
            "uint8_t",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
