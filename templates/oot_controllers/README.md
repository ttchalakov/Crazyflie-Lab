# Out-Of-Tree Controller Templates

Minimal, reusable starting points for writing a custom Crazyflie out-of-tree
(OOT) controller — the kind that **replaces the controller inside the stabilizer
loop** (`CONFIG_CONTROLLER_OOT=y`, implementing `controllerOutOfTreeInit/Test/
OutOfTree`). They are templates to copy, **not demos to fly**.

Complete, runnable control-loop demos live in `examples/<name>/controller`
instead; those are the place for a controller that does real work. When you start
a new one, copy a template here as the skeleton.

## What's here

- `zero_controller/` — the smallest valid OOT controller. Reads `state_t`,
  writes a `control_t` of all zeros. Compiles and flashes, produces no lift.
  Use it to verify the toolchain/API and as the blank slate for a new controller.

## The build contract

Each template is a self-contained OOT build root. Nothing is copied or generated;
the directory is built in place against the firmware submodule:

```
zero_controller/
  Makefile      # sets OOT=$(CURDIR), includes the firmware's tools/make/oot.mk
  app-config    # Kconfig fragment; CONFIG_CONTROLLER_OOT=y is the load-bearing line
  Kbuild        # obj-y += src/
  src/
    Kbuild      # obj-y += <name>.o
    <name>.c    # defines the controllerOutOfTree* symbols
```

The app-config also sets `# CONFIG_RADIO_ASSERT_ON_COM_PROBLEM is not set`: a
full radio queue or a bad checksum then drops the packet instead of asserting,
which would reboot the drone and cut the motors in flight.

Build it:

```bash
cf-lab build controller --app-dir templates/oot_controllers/zero_controller
```

`--example NAME` resolves to `examples/NAME/controller`. `docs/your-project.md`
shows the Makefile for a controller in your own repository.
