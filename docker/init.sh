#!/bin/sh
# One-shot setup the api service waits for: schema, taxonomy, and demo lots on first run.
set -eu

python -m alembic upgrade head

# artifacts/ is gitignored, so a fresh clone has no taxonomy. Only generate the
# synthetic fixture when the file is missing; a real dataset is left alone.
if [ ! -f artifacts/Dataset/fishora_dataset/metadata/taxonomy.csv ]; then
  python -m scripts.make_synthetic_taxonomy
fi
python -m scripts.seed_taxonomy

# The API upserts these at startup, but the demo lots below reference them and run first.
python -c "
from apps.main_api.config import MainSettings
from apps.main_api.db.lot_repository import SqlLandingPointRepository
from apps.main_api.db.session import session_factory
from apps.main_api.services.landing_points import seed_demo_landing_points

seed_demo_landing_points(SqlLandingPointRepository(session_factory(MainSettings())))
"

# seed_demo_lots replaces every demo_ row, which would wipe bids placed on them,
# so it only runs against an empty marketplace. FISHORA_SEED_DEMO_LOTS=0 skips it.
if [ "${FISHORA_SEED_DEMO_LOTS:-1}" = "1" ] && python - <<'EOF'
import sys

from sqlalchemy import func, select

from apps.main_api.config import MainSettings
from apps.main_api.db.models import Lot
from apps.main_api.db.session import session_factory

with session_factory(MainSettings())() as session:
    sys.exit(0 if session.scalar(select(func.count()).select_from(Lot)) == 0 else 1)
EOF
then
  python -m scripts.seed_demo_lots
fi
