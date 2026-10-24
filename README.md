# Zammoth's brain - source code

This is the source of the **brain**: the program that runs inside Zammoth.app at
`Contents/Resources/mixagent_daw_brain/`. It is the agent's half of the product - the mix
analysis, the offline render engine, the agent loop and the connection to the model - and the
app starts it as a separate process. It is licensed under the **GNU General Public License,
version 3** (the full text is `COPYING`), because it uses Pedalboard, which is GPLv3.

`SOURCE_VERSION` names the commit of Zammoth's own repository this was cut from and the app
build that commit made: this is the source of the brain shipped in that build.

## What is here

| Folder | What it is |
|---|---|
| `server/` | The audio engine: analysis (`mix_engine/analysis.py`), the offline render and its devices (`mix_engine/dsp.py`), the session file, the guardrails, the MCP tools, and the tests (`server/tests/`). |
| `orchestrator/` | The agent: the chat (`chat.py`), the run loop (`loop.py`, `agents.py`), the brain's socket server the app talks to (`daw_server.py`), sign-in and the licence check. |
| `daw/sidecar/` | How the brain is built into the program the app ships: `build.sh`, the PyInstaller recipe (`mixagent_daw_brain.spec`) and the entry point. |

## What is not here

- **`orchestrator/knowledge_content.py`, the knowledge book.** It is not part of the program:
  the shipped brain does not contain it, and fetches its text at run time from Zammoth's
  service when the user is signed in (`orchestrator/knowledge.py`). Without it the brain runs
  on a short fallback text.
- The app itself (C++), which is not GPL and is a separate program that talks to the brain
  over a local socket.

## Building it

On an Apple-silicon Mac with Python 3.14 (the version the shipped brain is built with):

```bash
python3 -m venv server/.venv
server/.venv/bin/pip install -r server/requirements.txt -r orchestrator/requirements.txt
daw/sidecar/build.sh
```

`build.sh` writes `daw/sidecar/dist/mixagent_daw_brain/`, the folder the app carries. The tests
run with `cd server && .venv/bin/python -m pytest`.

## Using your own build

Replace the folder in the app with yours, and sign the app again for your own Mac (macOS will
not open a signed app whose contents have changed):

```bash
rm -rf /Applications/Zammoth.app/Contents/Resources/mixagent_daw_brain
cp -R daw/sidecar/dist/mixagent_daw_brain /Applications/Zammoth.app/Contents/Resources/
codesign --force --deep --sign - /Applications/Zammoth.app
```

An update of Zammoth replaces the app, and your brain with it.
