# Environment setup

This project uses [uv](https://docs.astral.sh/uv/) to manage Python and its
dependencies.

## Quick start

Clone the repo, then from the repository root run:

**Windows (PowerShell)**

```powershell
.\SETUP\setup.ps1
```

**macOS / Linux**

```bash
./SETUP/setup.sh
```

The script installs uv if you don't have it, creates `.venv` with Python 3.12
(the version pinned in `.python-version`), and installs the exact package
versions from `requirements.lock`. It is safe to re-run at any time.

Then point your editor at the environment: in VS Code, open a notebook in
`Ressources/`, click the kernel picker in the top right, and choose the
interpreter at `.venv`. From a terminal, activate it with
`.\.venv\Scripts\Activate.ps1` (Windows) or `source .venv/bin/activate`
(macOS/Linux).

## Doing it by hand

The scripts are a convenience; these are the only two commands that matter:

```bash
uv venv                              # create .venv using the pinned Python
uv pip sync SETUP/requirements.lock  # install the locked versions
```

To install uv yourself, see the
[official instructions](https://docs.astral.sh/uv/getting-started/installation/):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh                                      # macOS / Linux
```

## GPU / CUDA

`requirements.lock` pins the CPU build of PyTorch, because that is the only
build that works everywhere. If you have an NVIDIA GPU, add a CUDA build after
setup — uv detects your driver and picks a matching wheel:

```bash
uv pip install torch --torch-backend=auto
```

Or pass `-Cuda` / `--cuda` to the setup script to do it in one step. Note that
`uv pip sync` will revert torch to the CPU build, so re-run this afterwards.

## Adding or changing a dependency

`requirements.txt` is the hand-edited list; `requirements.lock` is generated and
should never be edited by hand. After editing `requirements.txt`, regenerate the
lock and commit both files:

```bash
uv pip compile SETUP/requirements.txt --universal -o SETUP/requirements.lock
uv pip sync SETUP/requirements.lock
```

`--universal` makes the lock resolve for Windows, macOS, and Linux at once, so
one file serves every collaborator.

## Notes

- **tonic is not a dependency.** The notebook `git clone`s
  [neuromatch/tonic](https://github.com/neuromatch/tonic) and imports it from
  the checkout. Its requirements (`gym`, `termcolor`, and the scientific stack)
  are included in `requirements.txt` so the clone works straight away.
- **`gym` is pinned to 0.26.2** and prints a loud deprecation warning on import.
  That is expected — tonic depends on the pre-Gymnasium API. It works correctly
  with the NumPy 2.x in the lock file despite what the warning claims.
- **`legacy-colab-2021-requirements.txt`** records the original Colab pins from
  the upstream Neuromatch project. It is kept for provenance only and cannot be
  installed; see the header in that file.
