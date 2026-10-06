"""Stream specs: the STM32 log-block budget, period snapping and custom-stream validation."""
import pytest

from cf_lab.config import CustomStreamConfig, TelemetryConfig
from cf_lab.telemetry import schema


def _block_bytes(spec):
    return sum(schema.LOG_VAR_TYPES[cflib_type][0] for _, cflib_type in spec.log_variables)


@pytest.mark.parametrize(
    "spec",
    [schema.state_pos_spec(100.0), schema.state_att_spec(100.0), schema.ctrl_spec(100.0)],
    ids=lambda spec: spec.name,
)
def test_builtin_stm32_streams_fit_one_log_block(spec):
    assert 0 < _block_bytes(spec) <= schema.LOG_BLOCK_MAX_BYTES


@pytest.mark.parametrize(
    "variables, ok",
    [
        (("a.f0:float",) * 6 + ("a.h:uint16_t",), True),  # 26 bytes: exactly full
        (("a.f0:float",) * 6 + ("a.h:uint16_t", "a.b:uint8_t"), False),  # 27 bytes
        (("a.f0:float",) * 7, False),  # 28 bytes
        (("a.h:FP16",) * 13, True),  # FP16 is 2 bytes on the wire, not 4
        (("a.h:FP16",) * 14, False),
    ],
)
def test_custom_spec_enforces_26_byte_log_block(variables, ok):
    cfg = CustomStreamConfig(name="extra", period_ms=10, variables=variables)
    if ok:
        assert _block_bytes(schema.custom_spec(cfg)) <= schema.LOG_BLOCK_MAX_BYTES
    else:
        with pytest.raises(ValueError, match="26"):
            schema.custom_spec(cfg)


@pytest.mark.parametrize(
    "cfg, match",
    [
        (CustomStreamConfig(name="mocap", variables=("a.x:float",)), "reserved"),
        (CustomStreamConfig(name="events", variables=("a.x:float",)), "reserved"),
        (CustomStreamConfig(name="extra", variables=()), "no variables"),
        (CustomStreamConfig(name="extra", variables=("a.x",)), "unknown log type"),
        (CustomStreamConfig(name="extra", variables=("a.x:double",)), "unknown log type"),
        (CustomStreamConfig(name="extra", period_ms=15, variables=("a.x:float",)), "multiple of 10"),
        (CustomStreamConfig(name="extra", period_ms=0, variables=("a.x:float",)), "multiple of 10"),
        (CustomStreamConfig(name="extra", period_ms=2550, variables=("a.x:float",)), "multiple of 10"),
    ],
)
def test_custom_spec_rejects_invalid_config(cfg, match):
    with pytest.raises(ValueError, match=match):
        schema.custom_spec(cfg)


def test_custom_spec_maps_cflib_types_to_columns_and_keeps_period():
    cfg = CustomStreamConfig(
        name="extra",
        period_ms=40,
        variables=("motor.m1:uint16_t", "pm.vbat:float", "acc.h:FP16", "motor.req:int32_t"),
    )
    spec = schema.custom_spec(cfg)
    assert spec.clock == schema.CLOCK_STM32_MS
    assert spec.fields == (
        ("stm32_ms", "int64"),
        ("motor.m1", "uint16"),
        ("pm.vbat", "float32"),
        ("acc.h", "float32"),
        ("motor.req", "int32"),
    )
    assert spec.log_variables == (
        ("motor.m1", "uint16_t"),
        ("pm.vbat", "float"),
        ("acc.h", "FP16"),
        ("motor.req", "int32_t"),
    )
    assert dict(spec.meta)["period_ms"] == "40"


@pytest.mark.parametrize(
    "rate_hz, period_ms",
    [(100.0, 10), (50.0, 20), (30.0, 30), (1000.0, 10), (0.1, 2540), (1.0, 1000)],
)
def test_log_period_snaps_to_10ms_grid_within_bounds(rate_hz, period_ms):
    assert schema.log_period_ms(rate_hz) == period_ms


def test_build_specs_follows_config_toggles():
    cfg = TelemetryConfig(
        enabled=True,
        state_hz=0.0,
        controls_hz=50.0,
        setpoints=False,
        mocap=True,
        extpose=False,
        camera=True,
        custom=(CustomStreamConfig(name="tba", period_ms=10, variables=("tba.tick:uint32_t",)),),
    )
    specs = {spec.name: spec for spec in schema.build_specs(cfg, mocap_transport="vrpn")}
    assert set(specs) == {"ctrl", "tba", "mocap", "camera", "events"}
    assert dict(specs["ctrl"].meta)["period_ms"] == "20"
    assert specs["mocap"].clock == schema.CLOCK_ROS_STAMP_S


def test_build_specs_natnet_mocap_uses_frame_clock():
    specs = schema.build_specs(TelemetryConfig(enabled=True), mocap_transport="natnet")
    mocap = next(spec for spec in specs if spec.name == "mocap")
    assert mocap.clock == schema.CLOCK_NATNET_FRAME


def test_build_specs_rejects_a_bad_custom_stream_before_flight():
    cfg = TelemetryConfig(
        enabled=True, custom=(CustomStreamConfig(name="big", variables=("a.x:float",) * 7),)
    )
    with pytest.raises(ValueError, match="split into two streams"):
        schema.build_specs(cfg, mocap_transport="natnet")
