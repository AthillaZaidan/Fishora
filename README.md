# Fishora

Fishora is an AI catch-intelligence platform for Indonesian capture fisheries. A DINOv3 image
classifier identifies a landed fish, a human confirms or corrects it, and retrieval-augmented
generation turns the verified species into a grounded, source-cited commercial knowledge card in
Bahasa Indonesia.

The local runtime starts PostgreSQL in Docker and runs the CV service, the API and the web frontend
on the host. Only the database is containerised, so the CV service can reach a GPU directly when one
is available; it also runs on CPU at roughly a second per image, which is enough to demonstrate the
whole flow. To run every service in containers instead, see
[Or run everything in Docker](#or-run-everything-in-docker).

**Running this for the first time? Start with [Quick start](#quick-start).**

## Quick start

Everything below is copy-paste from the repository root. It assumes Python 3.11+, Node 20.9+ with
pnpm, and Docker. No GPU, no API key and no dataset are needed to get a working app.

**1. Pick the interpreter.** The venv layout differs by platform, so the rest of this guide uses
`$PY`:

```bash
python -m venv .venv
PY=.venv/Scripts/python.exe   # Windows (Git Bash / MSYS)
PY=.venv/bin/python           # macOS and Linux
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -e .
```

**2. Point at the database.** It has no default, and every command below needs it:

```bash
export FISHORA_DATABASE_URL=postgresql+psycopg://fishora:fishora@localhost:55432/fishora
```

**3. Create the schema and seed something to click through.** `artifacts/` is gitignored, so a fresh
clone has no taxonomy and no lots; these scripts stand in for the real dataset:

```bash
docker compose up -d db
"$PY" -m alembic upgrade head
"$PY" -m scripts.make_synthetic_taxonomy
"$PY" -m scripts.seed_taxonomy
"$PY" -m scripts.seed_demo_lots --reset
```

**4. Start the API** (leave it running):

```bash
"$PY" -m uvicorn apps.main_api.main:app --host 0.0.0.0 --port 8000
```

**5. Start the frontend** in a second terminal:

```bash
cd apps/frontend && pnpm install && pnpm dev --port 3111
```

Open **http://localhost:3111**.

### Signing in

Go to `/account` and pick an account from the dropdown. Both use the password `demo`, which the form
fills for you:

| Account | Role | What it can do |
| --- | --- | --- |
| Rian Setiawan | Operator | Photograph a catch, confirm the species, publish a lot, close and allocate an auction, print the QR card |
| Dewi Anggraini | Buyer | Browse the marketplace, set a buyer profile, bid by the kilogram, review a species after winning |

### A five-minute tour

1. **`/`** the landing page: the scroll descent, then the flow, then what a QR card carries.
2. **`/marketplace`** as Dewi: eleven species with real catch photography, filters by species, price
   and volume.
3. **`/marketplace/demo_lot_1`**: the knowledge panel, the market-signal reviews, and a bid bar.
4. **`/preferences`**: set a buyer profile and watch the match count move before you save.
5. **`/operator`** as Rian: the four-step publish flow.
6. **`/operator/lots`** as Rian: close an auction, allocate it, then **Buat QR** for the printable
   3:4 card.
7. **`/discover/demo_tenggiri-l-1`**: the public page a shopper reaches by scanning that card. No
   auth, no commercial data.

### Optional: AI species identification

The classifier needs the model export and two extra packages. It runs on CPU.

First, download the model export. It is not tracked in git (see `ai/results/` in
`.gitignore`), so a fresh clone has no weights. Get the archive from the shared
Google Drive folder and unpack it into the `ai/` directory as follows:

```bash
# 1. Download the archive from the shared folder:
#    https://drive.google.com/drive/folders/1PzeQ4AEnZdK48Nmq29az-2sImDiYubPk?usp=sharing
#    Save it as ai/fishora_export.zip (or whatever archive name the folder provides).

# 2. Extract it inside ai/ so the export lands at the default path:
cd ai
unzip fishora_export.zip
cd ..
```

The unpacked export must match the expected location
`ai/results/fishora_dinov3_large_frozen/export/` and contain `model_state_dict.pt`,
`inference_config.json` and `inference.py`. If the archive omits the
`results/fishora_dinov3_large_frozen/` part of the path, move the `export/` folder to
`ai/results/fishora_dinov3_large_frozen/` after extracting.

Then install the extra packages and start the CV service:

```bash
"$PY" -m pip install torch torchvision timm
FISHORA_CV_DEVICE=cpu "$PY" -m uvicorn apps.cv_service.main:app --host 0.0.0.0 --port 8001
```

With it running, `/operator` identifies a photograph instead of asking for the species. Note that the
shipped export has `abstain_threshold: 0.0`, so the API reports
`low_confidence_human_verification_required` for every prediction however high the score, and the
operator always confirms explicitly. That is the intended human-in-the-loop gate, not a fault.

### Optional: knowledge cards

Generated cards need three things, in this order:

1. `OPENCODE_GO_API_KEY` in `.env`.
2. The E5 weights in the local Hugging Face cache. The embedder is constructed with
   `local_files_only=True` so a request never triggers a download, which means fetching it once, on
   purpose:

```bash
"$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('intfloat/multilingual-e5-base')"
```

Miss any of those and the endpoint answers `502 knowledge retrieval is temporarily unavailable`.
Even with all three, cards come back empty with the limitation `No information available yet` until an
approved corpus is ingested: generation is fail-closed and will not assert anything it cannot cite.
That approval requires a human attestation and is deliberately not automated.

To load the corpus into a database (the 49 candidate chunks in `artifacts/knowledge_sources/`,
reviewed and approved by Jason Edward Salim on 2026-09-26):

```bash
export FISHORA_CORPUS_APPROVAL_KEY=$("$PY" -c "import secrets; print(secrets.token_hex(32))")
"$PY" -m scripts.corpus_pipeline approve --candidate-dir artifacts/knowledge_sources/candidates \
  --review-file artifacts/knowledge_sources/review/approval.json \
  --approved-dir artifacts/knowledge_sources/approved \
  --approval-manifest artifacts/knowledge_sources/approval-manifest.json \
  --reviewer "<your name>" --confirmation APPROVE
"$PY" -m scripts.corpus_pipeline ingest --approved-dir artifacts/knowledge_sources/approved \
  --approval-manifest artifacts/knowledge_sources/approval-manifest.json
```

### Knowledge sources

Every card claim cites one of 28 sources behind the 49 approved chunks, covering all 11 species.
Most come from three reference databases; the rest are peer-reviewed papers, each linked by DOI.

| Source | Used for | Where |
|---|---|---|
| **FishBase** (14 species pages) | Identity, physical traits, size, commercial status for every species | [fishbase.se](https://www.fishbase.se), one summary page per scientific name |
| **FAO** (4 documents) | Tuna and mackerel catalogue, Nile tilapia fact sheet, tilapia in Asia | [FAO Species Catalogue Vol. 2: Scombrids](https://www.fao.org/4/ac478e/ac478e00.htm), [Nile tilapia fact sheet](https://www.fao.org/fishery/docs/CDrom/aquaculture/I1129m/file/en/en_niletilapia.htm), [Tilapias in Asia and the Pacific](https://www.fao.org/4/y5728e/y5728e05.htm) |
| **NOAA Fisheries** | Yellowfin tuna taste, texture and marketing | [Pacific yellowfin tuna](https://www.fisheries.noaa.gov/species/pacific-yellowfin-tuna) |
| **Peer-reviewed papers** (9) | Processing, products and local market uses | Food Chemistry [10.1016/j.foodchem.2008.09.078](https://doi.org/10.1016/j.foodchem.2008.09.078); Food Research [10.26656/fr.2017.7(S3).12](https://doi.org/10.26656/fr.2017.7(S3).12); J. Bangladesh Agric. Univ. [10.5455/JBAU.86202](https://doi.org/10.5455/JBAU.86202); Coastal and Ocean Journal [10.29244/coj.5.1.1-8](https://doi.org/10.29244/coj.5.1.1-8); Jurnal Manajemen dan Agribisnis [10.17358/jma.22.2.210](https://doi.org/10.17358/jma.22.2.210); Jurnal IPTEKS PSP [10.20956/jipsp.v6i12.7801](https://doi.org/10.20956/jipsp.v6i12.7801); J. Pharmaceutical and Sciences [10.36490/journal-jps.com.v8i2.936](https://doi.org/10.36490/journal-jps.com.v8i2.936); Manfish Journal [10.31573/manfish.v2i3.489](https://doi.org/10.31573/manfish.v2i3.489); Marinade [10.31629/marinade.v5i02.4962](https://doi.org/10.31629/marinade.v5i02.4962) |

Protein per 100 g on the card comes from published food-composition tables, cited on the card
itself and listed in `apps/frontend/lib/nutrition.ts` (TKPI 2017 by Kemenkes RI, the Malaysian
Food Composition Database, the Thai FCD, and the ASEAN tables via FAO/INFOODS).

Gaps: no source covers substitute species for any fish, and taste/texture sources are missing for
six species, so those card fields stay empty rather than guessed
(`python -m evals.corpus_gaps` lists every empty cell).

## System at a Glance

| Service | Port | Runs on | Responsibility |
| --- | --- | --- | --- |
| Frontend (Next.js 16) | 3111 | Host, Node | Operator flow, marketplace, public QR profile |
| Main API (FastAPI) | 8000 | Host, Python | Identification, verification, knowledge cards |
| CV service (FastAPI) | 8001 | Host, Python + CUDA | DINOv3 species classification |
| PostgreSQL + pgvector | 55432 | Docker | Species, knowledge chunks and embeddings, predictions |

`scripts/run_local.sh` still defaults `FISHORA_FRONTEND_PORT` to 3000, but 3111 is the port used in
practice: 3000 is often taken by an unrelated app on this machine, and Playwright's `webServer` runs
`pnpm dev --port 3111` (`apps/frontend/playwright.config.ts`). The API's default CORS allow-list
covers both, so either port works from a browser without extra configuration.

The browser talks to the Main API directly, so two settings must agree or every request fails:
`NEXT_PUBLIC_API_BASE_URL` (where the frontend sends requests) and `FISHORA_CORS_ALLOW_ORIGINS`
(which browser origins the API accepts). `scripts/run_local.sh` derives both from the port variables,
so changing a port keeps them in step automatically.

```text
Browser ──HTTP──> Frontend :3111
   │
   └────HTTP────> Main API :8000 ──> CV service :8001 ──> DINOv3 (CUDA)
                      │
                      ├──> PostgreSQL + pgvector :55432
                      └──> OpenCode Go (knowledge card generation)
```

## API Surface

Identification and knowledge:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness plus `taxonomy_seeded` |
| POST | `/api/v1/fish/identify` | Multipart image upload, returns a prediction and ranked candidates |
| POST | `/api/v1/fish/verify` | Human confirms or corrects the predicted species |
| POST | `/api/v1/fish/manual` | Operator declares the species themselves, used when the CV service is down |
| GET | `/api/v1/predictions/{id}/knowledge` | Grounded knowledge card for a **verified** prediction |

Commerce, buyers, and session:

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/v1/lots` | Publish a verified catch as `lot_count` lots of `quantity_kg` each. Optional `auction_minutes` |
| GET | `/api/v1/lots` | Public list with filters. `mine=1` scopes it to the signed-in operator |
| GET | `/api/v1/lots/{id}` | One lot |
| POST | `/api/v1/lots/{id}/bids` | Place a bid (buyer session) |
| GET | `/api/v1/lots/{id}/bids` | Bid history for a lot |
| POST | `/api/v1/lots/{id}/close` | Listing operator closes the auction |
| POST | `/api/v1/lots/{id}/allocate` | Listing operator allocates to the winning bid |
| PUT | `/api/v1/buyers/{id}/preferences` | Buyer preference profile |
| GET | `/api/v1/buyers/{id}/recommendations` | Explainable matches for a buyer |
| GET | `/api/v1/discover/{public_slug}` | Public consumer profile from the publish-time snapshot |
| POST | `/api/v1/auth/login`, `/api/v1/auth/logout`, GET `/api/v1/auth/me` | Role-based demo session |

Interactive docs: `http://localhost:8000/docs`.

`GET /health` returns `{"status": "ok", "taxonomy_seeded": bool}`. `taxonomy_seeded` is false when the
species table is empty, which is a distinct failure mode from the API being down: the API answers 200,
but every identification and manual declaration fails on species resolution.

`POST /api/v1/fish/manual` takes multipart `file` plus a `species_id` form field and creates a
verified prediction directly, so a CV outage cannot strand a catch. It records
`model_version="manual-entry"` and `confidence=0.0` so the audit trail never implies the model agreed.

A knowledge card is only issued for a verified prediction. Requesting one for a pending prediction
returns `409` by design: an AI guess must not become a public commercial record.

`POST /api/v1/lots` splits one catch into `lot_count` lots (1 to `MAX_LOT_COUNT`, 50) of
`quantity_kg` each; every lot is its own auction with its own winner, and the response is the list.
`auction_minutes` is optional and must be one of `AUCTION_MINUTE_OPTIONS` (30, 60, 120, 180) in
`apps/main_api/services/lots.py`. Omitting it keeps the 60-minute default.

## Prerequisites

Required:

- Python 3.11 or newer
- Node.js 20.9 or newer (22.x recommended) and pnpm 10
- Docker with Docker Compose

Optional, and only for AI species identification:

- A model export containing `model_state_dict.pt`, `inference_config.json` and `inference.py`. The
  default path is `ai/results/fishora_dinov3_large_frozen/export`. It is not in git, so download the
  archive from the shared Google Drive folder and unpack it into `ai/` — see
  [AI species identification](#ai-species-identification).
- An NVIDIA GPU with a CUDA driver. Not required: the classifier runs on CPU at about a second per
  image. Without the export at all, the operator names the species by hand, which is a designed path
  rather than a workaround, and nothing downstream is affected.

Optional, and only for generated knowledge cards:

- `OPENCODE_GO_API_KEY` in `.env`, plus the embedding model cached locally. See
  [Knowledge cards](#knowledge-cards).

## Setup

### 1. Python environment

Create one virtual environment and install CUDA-enabled PyTorch before installing Fishora:

```bash
python3.11 -m venv .venv

# The venv layout differs by platform. Set this once and reuse it everywhere below.
PY=.venv/Scripts/python.exe   # Windows (Git Bash / MSYS)
PY=.venv/bin/python           # macOS and Linux

"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -e '.[cv,dev]'
```

`run_local.sh` detects either layout on its own, so the run command below is the same either way.

The torch line is only needed for the CV service. Skip it to run everything else; identification
then returns 503 and the operator picks the species by hand.

Verify that PyTorch can access the GPU:

```bash
.venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

Cache the embedding model once while internet access is available. Runtime loading is offline-only.

```bash
.venv/bin/python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("intfloat/multilingual-e5-base")'
```

### 2. Frontend dependencies

```bash
cd apps/frontend
pnpm install
cd ../..
```

### 3. Configuration

Defaults are ready for local use. Create `.env` only to override them or to supply an OpenCode Go
API key for generated knowledge cards:

```bash
cp .env.example .env
```

Set `OPENCODE_GO_API_KEY` in `.env`. A verified species gets its card from the writer-critic
workflow (`apps/main_api/services/workflow.py`): one writer call, then a claim verifier that checks
every claim against the chunk it cites. Both share one OpenCode Go client configured with
`FISHORA_OPENCODE_GO_MODEL=gpt-6-luna`. The earlier four-expert graph is kept for comparison in
`evals/iteration2.py`. Without the key
everything still starts, and identification and verification work normally; only knowledge card
generation fails, with a `502`.

The key is not sufficient on its own. Retrieval runs before generation, so a knowledge card also
needs the embedding stack (`sentence-transformers` and `langchain-huggingface`, core dependencies of
`pip install -e .`, plus torch and numpy, which sentence-transformers pulls in) **and** the E5
weights already in the local Hugging Face cache. The embedder is constructed with
`local_files_only=True` so a request never triggers a download, which means the model has to be
fetched once, deliberately:

```bash
"$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('intfloat/multilingual-e5-base')"
```

Skip that and the endpoint answers `502 knowledge retrieval is temporarily unavailable`, which is
the retrieval stack missing rather than anything wrong with your API key.

**Tests that construct `MainSettings` must pass `_env_file=None`.** `MainSettings` sets
`env_file=".env"`, so on a machine with a real `.env` the file beats `monkeypatch.delenv`, a test
pinning defaults asserts that developer's values instead, and if the field is the API key pytest
prints the live secret in its assertion diff.

Never commit `.env` or API keys.

## Run

Start the whole stack:

```bash
bash scripts/run_local.sh
```

The script verifies CUDA, starts PostgreSQL, applies migrations, seeds taxonomy data, then launches
the CV service, the API, and the frontend. It prints the resolved URLs, the API base the frontend
will call, and the CORS allow-list, so a mismatch is visible before you open a browser. DINOv3 runs
on CUDA while multilingual E5 embeddings run on CPU.

Two preconditions are worth checking before blaming the code:

- `scripts/seed_taxonomy` reads `artifacts/Dataset/fishora_dataset/metadata/taxonomy.csv`. Without
  that file the species table stays empty, `GET /health` reports `taxonomy_seeded: false`, and every
  identification and manual declaration fails on species resolution.
- The CV service needs PyTorch. In an environment where torch is not installed it cannot start at
  all, so any flow that needs AI identification has to go through `POST /api/v1/fish/manual`
  instead. Verification, publication, bidding, and knowledge snapshots are unaffected.

### What the seed scripts do

The command sequence lives in [Quick start](#quick-start); this is what those three scripts are for.

`make_synthetic_taxonomy` stamps every row `synthetic-dev-fixture` and refuses to overwrite a file
that does not look synthetic, so restoring the real dataset later is safe. `seed_demo_lots` writes
`demo_`-prefixed rows only, attaches an illustrative knowledge card to each lot so the knowledge
panel and buyer matching have something to work on, and allocates one Nila lot to `buyer_dewi` so
the review flow is reachable.

`--reset` clears every lot first, including ones published by hand and the one the PRD walkthrough
publishes on each `playwright test` run. Without it a development database collects another
allocated Tenggiri every time the e2e suite runs.

None of these values are authoritative. Replace them with the real dataset before publishing any
claim about a species.

### Or start everything at once

```bash
FISHORA_FRONTEND_PORT=3111 bash scripts/run_local.sh
```

The script verifies CUDA, starts PostgreSQL, applies migrations, seeds taxonomy, then launches the
CV service, the API and the frontend, printing the resolved URLs, the API base compiled into the
browser bundle, and the CORS allow-list, so a mismatch is visible before you open a browser. It
needs the CV stack installed; use the manual sequence above if torch is missing.

Demo logins are `rian` / `demo` (operator) and `dewi` / `demo` (buyer).

Stop with `Ctrl+C`. All three processes are terminated; PostgreSQL stays up in Docker:

```bash
docker compose down
```

### Or run everything in Docker

Needs only Docker and a `.env` (copy `.env.example`); no Python, Node or pnpm on the host:

```bash
docker compose up --build
```

Open the frontend on `FISHORA_FRONTEND_PORT` from `.env` (3111 if unset; keep
`FISHORA_CORS_ALLOW_ORIGINS` in step with it). Compose reads the same `.env` as
`run_local.sh`, both for the services and for the published ports, and only swaps the host
addresses (`localhost:55432`, `localhost:8001`) for the service names.

| Service | What it does |
|---|---|
| `db` | PostgreSQL + pgvector, same volume and port (55432) as the host workflow |
| `init` | Migrations, taxonomy (the synthetic fixture if `artifacts/` has none), demo lots into an empty database, then exits |
| `api` | Main API on `FISHORA_MAIN_API_PORT`, E5 weights baked into the image, `data/` and `reports/` mounted |
| `frontend` | `next build` + `next start` on `FISHORA_FRONTEND_PORT`. `NEXT_PUBLIC_API_BASE_URL` is a build argument, so rebuild after changing it |
| `cv` | Opt-in: `COMPOSE_PROFILES=cv` in `.env`. Needs the export under `ai/` and an NVIDIA GPU visible to Docker |

Without the `cv` profile, identification returns 503 and the operator names the species by hand, as
in the host workflow. Demo lots are seeded only when the database has no lots; set
`FISHORA_SEED_DEMO_LOTS=0` to skip them. Stop with `docker compose down`; add `-v` to also drop the
database volume.

### Running a subset

```bash
# Backend and CV only, no Node required
FISHORA_SKIP_FRONTEND=1 bash scripts/run_local.sh

# Frontend alone, against an API already running elsewhere
cd apps/frontend && pnpm dev
```

Run standalone, the frontend defaults to `http://localhost:8000` for the API. Point it elsewhere with
`NEXT_PUBLIC_API_BASE_URL`, and remember to add that frontend's origin to
`FISHORA_CORS_ALLOW_ORIGINS` on the API side.

### Avoiding port conflicts

Override any port; the derived URLs and CORS list follow:

```bash
FISHORA_FRONTEND_PORT=3111 FISHORA_MAIN_API_PORT=8080 bash scripts/run_local.sh
```

3000 is the script's default and 3111 is the port that is actually free on this machine, so passing
`FISHORA_FRONTEND_PORT=3111` is the normal case rather than the exception.

### Testing on a real phone

The dev server binds `0.0.0.0`, so a phone on the same network can open
`http://<your-machine-ip>:3111`. DESIGN.md treats mobile as the primary target, and layout at 390px
cannot be judged from a desktop browser. For the phone to reach the API, use your machine's address
on both sides:

```bash
FISHORA_FRONTEND_PORT=3111 \
FISHORA_CORS_ALLOW_ORIGINS="http://192.168.1.20:3111" \
NEXT_PUBLIC_API_BASE_URL="http://192.168.1.20:8000" \
bash scripts/run_local.sh
```

## Verify the Services

```bash
curl -fsS http://localhost:8000/health                       # API: status and taxonomy_seeded
curl -fsS http://localhost:8001/health                       # CV service: status and model_version
curl -fsS http://localhost:8000/openapi.json | head -c 200   # API is serving
```

`{"status":"ok","taxonomy_seeded":false}` means the API is healthy but unseeded. Fix the taxonomy
file before debugging anything downstream of species resolution.

Confirm the browser is actually allowed to call the API. An empty `access-control-allow-origin` here
is the cause of nearly every "the frontend cannot reach the backend" report:

```bash
curl -si -X OPTIONS http://localhost:8000/api/v1/fish/identify \
  -H 'Origin: http://localhost:3111' \
  -H 'Access-Control-Request-Method: POST' | grep -i access-control-allow-origin
```

The default allow-list (`DEFAULT_CORS_ALLOW_ORIGINS` in `apps/main_api/config.py`) covers
`localhost` and `127.0.0.1` on both 3000 and 3111, with `allow_credentials=True` so the session
cookie survives the round trip.

Then open `http://localhost:3111`.

## Run Tests

> **Most test suites are not tracked in git.** `tests/`, every `*.test.ts(x)`, `apps/frontend/e2e/`,
> `playwright.config.ts` and `vitest.config.ts` are gitignored, so a fresh clone has none of them and
> the commands below find nothing to run. They work on a checkout that already has them on disk.
> The RAG evaluation suite in `evals/tests` is tracked; see [Evaluate](#evaluate).

Same `$PY` and `FISHORA_DATABASE_URL` as above.

### Backend

```bash
"$PY" -m pytest tests -q
```

Expect **300 passed, 4 failed, 6 skipped**. The four failures are all
`tests/main_api/test_embeddings.py` raising `ModuleNotFoundError: No module named 'numpy'`: the torch
and sentence-transformers stack is not installed in this venv. They are environmental, not product
failures. Anything else failing is a real regression.

Subsets, when the database is not up:

```bash
"$PY" -m pytest -m "not integration and not real_artifact" -q
```

### Frontend

```bash
cd apps/frontend
pnpm test     # Vitest: 35 files, 183 tests
pnpm lint     # 0 errors, 0 warnings
pnpm build    # type check plus production build
```

### End to end

Playwright drives a real browser against the running stack, so start the API and the frontend first.
Three viewport projects run: `phone` (Pixel 7), `phone-390` (the narrowest width DESIGN.md commits
to) and `desktop`.

```bash
cd apps/frontend
npx playwright test                      # 94 passed, 14 skipped
npx playwright test --project=phone-390  # the narrowest layout alone
```

The skips are honest: the PRD walkthrough runs on one project rather than three because it publishes
real rows, and the identification specs skip when the CV service is unreachable. `mvp.spec.ts` is
the eleven-step PRD 27 walkthrough and does exercise the live backend end to end.

## Evaluate

Two evaluation suites feed one dashboard. Both run offline and need no database.

| Suite | Measures | Needs |
| --- | --- | --- |
| `evals/` | Knowledge-card RAG: retrieval, claim grounding, the card pipeline, real-LLM cost | The E5 weights in the local cache. `OPENCODE_GO_API_KEY` only for the cost runs |
| `evaluation/cv/` | Species identification: six backbones on clean, corrupted, background, field and out-of-distribution photos, plus CPU latency | Nothing to view the committed results. Test photos and model exports to regenerate them |

### RAG suite and dashboard

One command runs the test suite with coverage, the retrieval, grounding and pipeline evaluations,
the code probes, the CV import, the weakness registry and the dashboard:

```bash
HF_HUB_OFFLINE=1 "$PY" -m scripts.quality --label baseline
```

It writes `reports/<label>/` (JSON artifacts, JUnit XML, coverage) and `reports/dashboard.html`.
`reports/` is generated and gitignored; the recorded baseline, including a copy of the dashboard,
lives in `evals/results/baseline/`.

The baseline code (tag `checkpoint-1-baseline`) gives **35 passed, 15 failed**. The failures were
deliberate: each one is a weakness a fix had to close, and the dashboard lists them with their
computed evidence. The recorded iteration-1 run passes **all 71**
(`evals/results/iteration-1/tests.json`) and the iteration-2 run **all 84**
(`evals/results/iteration-2/tests.json`). The test layers can also run alone:

```bash
HF_HUB_OFFLINE=1 "$PY" -m pytest evals/tests -m unit -q        # no model load, about 2 s
HF_HUB_OFFLINE=1 "$PY" -m pytest evals/tests -q                # all layers, about 40 s
```

`evals/README.md` describes each layer and what the measurements can and cannot show.

### Real-LLM cost and model comparison

These call OpenCode Go and spend from its per-model allowance ($3 per 5 hours for each Luna model).
A full cost run costs about $0.07; the comparison runs 11 cards on each of six models.

```bash
"$PY" -m evals.cost_eval --label baseline --repeat 2
"$PY" -m evals.model_compare --label baseline
"$PY" -m evals.dashboard
```

Both the application client and the cost run send the `x-opencode-session` header the gateway
requires. The cost run drives the application client itself, so a missing header shows up as failed
cards. Its paths are `agent` (the graded card the product publishes) and `one_call` (the ungraded
one-call generator, kept only as a reference for W26).

The claim verifier experiment re-reads the claims of three agent-only cost runs (labels
`iter2-diagnosis`, `iter2-claims-2`, `iter2-claims-3`, about $0.01 each; archived in
`evals/results/iteration-2/claim_runs/`, copy them to `reports/<label>/`) and needs no key itself:

```bash
"$PY" -m evals.cost_eval --label iter2-claims-2 --paths agent --repeat 1   # one of the three runs
HF_HUB_OFFLINE=1 "$PY" -m evals.experiment_verifier --label current
```

It also needs the two compact NLI models in the local cache
(`MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` and `-L12-`); the product does not use them.

### Species identification

The results for six linear-probe backbones are committed in `evaluation/cv/results/`, so the
dashboard shows them without running anything. To regenerate them:

1. Install the suite's extra dependencies: `"$PY" -m pip install -e '.[cv,cv-eval]'`
   (`rembg` in that extra is only needed for step 4).
2. Fetch the external test photos: `"$PY" -m evaluation.cv.prep.fetch_external_data`. Every image
   is recorded in `evaluation/cv/data/attribution.csv`.
3. Screen them with `"$PY" -m evaluation.cv.prep.contact_sheet`. The screening is locked in
   `evaluation/cv/protocol/screening.lock`; do not edit it once results exist.
4. Generate fish masks for the background slices: `"$PY" -m evaluation.cv.prep.make_masks`.
5. Train or download the linear-probe exports with `evaluation/cv/kaggle/fishora_linear_probe.ipynb`
   (`evaluation/cv/kaggle/pack_kit.py` builds the Kaggle upload), and unpack them under `LP_DIR`.
6. Run every model, the CPU latency bench and the comparison:

```bash
PY="$PY" bash evaluation/cv/run_all.sh                    # everything
SKIP_EXISTING=1 PY="$PY" bash evaluation/cv/run_all.sh    # keep runs that already have predictions.csv
```

`evaluation/cv/data/` and `evaluation/cv/kaggle_outputs/` (about 9 GB with weights) are gitignored.
One model can also be run on its own:

```bash
"$PY" -m evaluation.cv.cv_suite --run lp_vit_l --export <export_dir>
"$PY" -m evaluation.cv.report lp_vit_l
```

### Iteration 2: architecture x model experiment

```bash
HF_HUB_OFFLINE=1 "$PY" -m evals.iteration2_gold --label iteration-2
HF_HUB_OFFLINE=1 "$PY" -m evals.iteration2 --label iteration-2 --judge glm-5.3-flash --repeat 1 --critic-variants e5
```

The first grades the 221 locked gold claims (`evals/protocol/claims_gold.csv`, AI-assisted
annotation reviewed by the team) with every candidate critic and judge. The second generates cards
for {four-expert graph, writer-critic} x {gpt-5.6-luna, gpt-6-luna} and scores their claims with the
judge. Results are recorded in `evals/results/iteration-2/`.

### Compare a change against the baseline

After changing the RAG pipeline or the classifier, run the same suite under a new label:

```bash
HF_HUB_OFFLINE=1 "$PY" -m scripts.quality --label current
```

The dashboard then shows each target metric as baseline, target and current, and
`reports/comparison.json` lists every test that was fixed or regressed.

### Iteration documentation

Each iteration is recorded as findings, then fixes, then a re-test, in `evals/results/<iteration>/`.
The documents are generated by `"$PY" -m evals.iteration --name <iteration> --baseline <run> --current <run>`
from the run artifacts, so their numbers are never typed by hand. Iteration 2 is measured against
iteration 1's archived artifacts (copy `evals/results/iteration-1/*.json` to `reports/iteration-1/`).

| Document | What it holds |
| --- | --- |
| [`evals/results/iteration-2/CYCLES.md`](evals/results/iteration-2/CYCLES.md) | **Latest.** Every card that reaches a buyer passes the claim critic (W19); card runs leave a stage trace (W12); the claim verifier re-tested on real expert claims (W26, W29, R2) |
| [`evals/results/iteration-2/REPORT.md`](evals/results/iteration-2/REPORT.md) | Every iteration-2 fix (F19, F20 product; E8 to E13 evaluation), the verifier experiment, what is still open |
| [`evals/datasets/grounding_real_claims.json`](evals/datasets/grounding_real_claims.json) | 345 labelled real claim atoms and 23 fresh traps; labels by an AI assistant, **pending human review** |
| [`evals/results/iteration-1/CYCLES.md`](evals/results/iteration-1/CYCLES.md) | The three main RAG cycles of iteration 1, each with the finding, the product and evaluation fixes, and the re-test |
| [`evals/results/iteration-1/REPORT.md`](evals/results/iteration-1/REPORT.md) | Every fix (F1 to F18 product, E1 to E7 evaluation) with its source finding, files, tests and metrics; the experiments; what is still open |
| `evals/results/iteration-1/dashboard.html` | The dashboard, baseline against iteration 1 |
| `evals/results/iteration-1/iteration.json` | The same record for tools |

Every fix names the finding it came from: W* from the baseline registry in `evals/findings.py`, R* from
the review critique. A change without a source finding does not go in.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Browser console shows a CORS error | The frontend's origin is not in `FISHORA_CORS_ALLOW_ORIGINS`. Run the OPTIONS check above. Changing `FISHORA_FRONTEND_PORT` without restarting the API is the usual cause |
| `pnpm not found`, backend starts anyway | Install pnpm, or keep going with `FISHORA_SKIP_FRONTEND=1` |
| `Frontend dependencies missing` | `cd apps/frontend && pnpm install` |
| `PyTorch is not installed in .venv` | The `cv` extra was never installed. The script prints the exact two commands to run |
| `cannot see a CUDA device` | Torch is installed without CUDA support. Reinstall from the cu128 index |
| `Missing .venv` | No virtualenv found at `.venv`. The script accepts both `bin/` (POSIX) and `Scripts/` (Windows) layouts |
| `/health` reports `taxonomy_seeded: false` | The species table is empty. `scripts/seed_taxonomy` needs `artifacts/Dataset/fishora_dataset/metadata/taxonomy.csv` |
| Identification returns 503 | The CV service is down or was never started (no torch). Use `POST /api/v1/fish/manual` to keep publishing |
| Knowledge card returns 502 | `OPENCODE_GO_API_KEY` is blank or OpenCode Go is unreachable. Identification and verification are unaffected |
| Knowledge card returns 409 | The prediction is still `pending`. Verify the species first |
| `EADDRINUSE` on startup | A previous run is still bound. Override the port, or stop the old process |
| Frontend loads but every request fails | Check the `API base` line the run script prints. It is compiled into the browser bundle, so it must be reachable *from the browser*, never an internal hostname |

## Repository Layout

```text
apps/
  frontend/       Next.js 16 app: App Router, Tailwind v4, design tokens
  main_api/       FastAPI: identification, verification, RAG knowledge cards
  cv_service/     FastAPI: DINOv3 inference
  common/         Shared image validation
ai/training/      PRD.md, the product requirements document
alembic/          Schema migrations
scripts/          run_local.sh, taxonomy and demo seeding, corpus pipeline, quality.py
evals/            RAG evaluation, its test suite (evals/tests), baseline and dashboard generator
evaluation/cv/    Species-identification evaluation suite and committed results
artifacts/        Knowledge corpus and model export (gitignored)
reports/          Generated evaluation output and dashboard (gitignored)

Not tracked, present only in a working checkout: the other test suites (tests/, *.test.tsx, e2e/),
DESIGN.md, and the handoff and planning notes.
```
