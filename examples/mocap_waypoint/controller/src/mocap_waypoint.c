// Default-firmware build: there is no out-of-tree app or controller here. This
// translation unit exists only so the out-of-tree build emits the built-in.o the
// firmware links against; it adds no behavior (CONFIG_APP_ENABLE is unset, so the
// firmware never calls an appMain). The waypoint demo runs entirely on the built-in
// PID position controller, driven from the host via streamed position setpoints.
typedef int mocap_waypoint_no_app;
