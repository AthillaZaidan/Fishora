"""Demo landing points used by the operator form and geo filter."""

from apps.main_api.contracts import LandingPointRecord

DEMO_LANDING_POINTS = (
    LandingPointRecord(id="lp_muara_angke", name="PPI Muara Angke", latitude=-6.104, longitude=106.792),
    LandingPointRecord(id="lp_cilacap", name="TPI Cilacap", latitude=-7.732, longitude=109.015),
    LandingPointRecord(id="lp_karangsong", name="PPI Karangsong", latitude=-6.305, longitude=108.320),
)


def seed_demo_landing_points(repo) -> None:
    for point in DEMO_LANDING_POINTS:
        repo.upsert(point)


def list_landing_points(repo) -> list[LandingPointRecord]:
    """Every landing point an operator can publish from, in seed order.

    Without a repository (a test app with no database) the seed list is the
    answer, since that is exactly what production seeds at startup.
    """
    points = list(repo.all()) if repo is not None else list(DEMO_LANDING_POINTS)
    order = {point.id: index for index, point in enumerate(DEMO_LANDING_POINTS)}
    return sorted(points, key=lambda point: (order.get(point.id, len(order)), point.name))
