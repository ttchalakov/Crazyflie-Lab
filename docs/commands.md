# Commands

`cf-lab <command> --help` shows every option.

| Command | Does |
|---|---|
| `cf-lab build controller\|gap8\|esp` | Build one processor's firmware |
| `cf-lab flash controller\|gap8\|esp` | Build, then flash it over the radio |
| `cf-lab compile-model` | Compile a PyTorch model into a GAP8 app: NEMO quantizes it, DORY generates the C |
| `cf-lab view` | Show the AI-deck camera stream in a window |
| `cf-lab find-deck` | Print the address of the drone's AI-deck, found by name on the network |
| `cf-lab reboot` | Power-cycle the drone and wait until it is healthy |
| `cf-lab shutdown` | Power the drone fully off; only its power button brings it back |
| `cf-lab preflight` | Read-only checks before flying |
| `cf-lab config-value KEY` | Print one value from the config, for example `radio.uri` |

Commands that talk to the drone read its address, `radio.uri`, from `--config`
(default `configs/drone.yaml` in the current directory); `flash`, `reboot`,
`shutdown` and `find-deck` also take `--uri`.

## Build and flash options

| Option | Meaning |
|---|---|
| `--example NAME` | An example in the cf-lab checkout (`controller` and `gap8`) |
| `--app-dir DIR` | Your app's directory, which holds its Makefile (`controller` and `gap8`) |
| `--no-build` | Flash the last build |
| `--artifact FILE` | Flash this file instead |
| `--cold-boot` | `gap8` and `esp` only: power-cycle the drone first, to recover a wedged deck |

`cf-lab build gap8` and `cf-lab flash gap8` also take `--make-args "VAR=VALUE ..."`
for the app's make variables, and build the deck's WiFi settings from the
`camera:` section of the config into apps that use them. `--wifi-mode`,
`--ssid`, `--key`, `--transport` and `--channel` override those settings for one
build.

The GAP8, ESP32, NEMO and DORY builds run in Docker. `cf-lab` finds the SDK
checkout from `$CF_LAB` if it is set, otherwise from where the package is
installed.
