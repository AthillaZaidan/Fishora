"""Lot publication freezes the critic-graded card when one exists (W19)."""

from decimal import Decimal

import pytest

from apps.main_api.errors import InvalidLot
from apps.main_api.services.lots import LotService
from evals.fakes import InMemoryJobRepository, InMemoryPredictionRepository

GRADED = {"common_name": "Nila", "taste": "Rasa ringan", "sources": [{"source_id": "fao_en_niletilapia"}]}


class _Lots:
    def __init__(self):
        self.rows = {}

    def get_by_prediction(self, prediction_id):
        return next((l for l in self.rows.values() if l.prediction_id == prediction_id), None)

    def create_many(self, lots):
        self.rows.update({lot.id: lot for lot in lots})
        return lots


class _SyncKnowledge:
    def __init__(self):
        self.calls = 0

    def get_for_prediction(self, prediction_id):
        self.calls += 1
        raise AssertionError("the ungraded sync path must not run when a graded card exists")


def _publish(jobs, knowledge, **kwargs):
    predictions = InMemoryPredictionRepository()
    predictions.create("p1", "memory://p1", "species_nila", 0.9, [], "v")
    predictions.verify("p1", "species_nila", "confirmed")
    service = LotService(predictions, _Lots(), knowledge_service=knowledge, job_repo=jobs)
    return service.publish(prediction_id="p1", operator_id="op", quantity_kg=Decimal("10"),
                           starting_price_per_kg=Decimal("20000"), size_category="M", landing_point_id="lp1",
                           **kwargs)


def test_published_snapshot_is_the_graded_job_card():
    jobs = InMemoryJobRepository()
    jobs.create("p1", "p1", "species_nila")
    jobs.update("p1", status="completed", final_card=GRADED)
    knowledge = _SyncKnowledge()
    [lot] = _publish(jobs, knowledge)
    assert knowledge.calls == 0
    assert lot.knowledge_snapshot["taste"] == GRADED["taste"]
    assert lot.knowledge_snapshot["sources"] == GRADED["sources"]


def test_empty_fields_of_the_graded_card_come_from_the_reference_notes():
    from apps.main_api.services.reference_cards import REFERENCE

    jobs = InMemoryJobRepository()
    jobs.create("p1", "p1", "species_nila")
    jobs.update("p1", status="completed", final_card=GRADED)
    [lot] = _publish(jobs, _SyncKnowledge())
    card = lot.knowledge_snapshot
    # The sourced field is kept; only the empty ones are filled, and labelled.
    assert card["taste"] == GRADED["taste"]
    assert card["texture"] == REFERENCE["nila"][2]
    assert card["processing_methods"] == REFERENCE["nila"][3]
    assert any("reference notes" in item and "taste" not in item for item in card["limitations"])


def test_graded_card_for_another_species_is_not_published():
    jobs = InMemoryJobRepository()
    jobs.create("p1", "p1", "species_mujair")  # graded before a correction
    jobs.update("p1", status="completed", final_card=GRADED)

    class Fallback:
        calls = 0

        def get_for_prediction(self, prediction_id):
            Fallback.calls += 1
            raise RuntimeError("generation down")

    [lot] = _publish(jobs, Fallback())
    assert Fallback.calls == 1 and lot.knowledge_snapshot is None


def _graded_jobs():
    jobs = InMemoryJobRepository()
    jobs.create("p1", "p1", "species_nila")
    jobs.update("p1", status="completed", final_card=GRADED)
    return jobs


def test_a_catch_publishes_as_numbered_lots_sharing_one_card():
    lots = _publish(_graded_jobs(), _SyncKnowledge(), lot_count=3, auction_minutes=30)
    assert [(lot.batch_index, lot.batch_size) for lot in lots] == [(1, 3), (2, 3), (3, 3)]
    assert len({lot.id for lot in lots}) == len({lot.public_slug for lot in lots}) == 3
    card = lots[0].knowledge_snapshot
    assert card["taste"] == GRADED["taste"]
    assert all(lot.quantity_kg == Decimal("10") and lot.knowledge_snapshot == card for lot in lots)
    assert all((lot.auction_ends_at - lot.auction_starts_at).total_seconds() == 30 * 60 for lot in lots)


@pytest.mark.parametrize("kwargs", [{"auction_minutes": 45}, {"lot_count": 0}, {"lot_count": 51}])
def test_out_of_range_batch_or_duration_is_refused(kwargs):
    with pytest.raises(InvalidLot):
        _publish(_graded_jobs(), _SyncKnowledge(), **kwargs)
