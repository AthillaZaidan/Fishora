"""Lot publication freezes the critic-graded card when one exists (W19)."""

from decimal import Decimal

from apps.main_api.services.lots import LotService
from evals.fakes import InMemoryJobRepository, InMemoryPredictionRepository

GRADED = {"common_name": "Nila", "taste": "Rasa ringan", "sources": [{"source_id": "fao_en_niletilapia"}]}


class _Lots:
    def __init__(self):
        self.rows = {}

    def get_by_prediction(self, prediction_id):
        return next((l for l in self.rows.values() if l.prediction_id == prediction_id), None)

    def create(self, lot):
        self.rows[lot.id] = lot
        return lot


class _SyncKnowledge:
    def __init__(self):
        self.calls = 0

    def get_for_prediction(self, prediction_id):
        self.calls += 1
        raise AssertionError("the ungraded sync path must not run when a graded card exists")


def _publish(jobs, knowledge):
    predictions = InMemoryPredictionRepository()
    predictions.create("p1", "memory://p1", "species_nila", 0.9, [], "v")
    predictions.verify("p1", "species_nila", "confirmed")
    service = LotService(predictions, _Lots(), knowledge_service=knowledge, job_repo=jobs)
    return service.publish(prediction_id="p1", operator_id="op", quantity_kg=Decimal("10"),
                           starting_price_per_kg=Decimal("20000"), size_category="M", landing_point_id="lp1")


def test_published_snapshot_is_the_graded_job_card():
    jobs = InMemoryJobRepository()
    jobs.create("p1", "p1", "species_nila")
    jobs.update("p1", status="completed", final_card=GRADED)
    knowledge = _SyncKnowledge()
    lot = _publish(jobs, knowledge)
    assert lot.knowledge_snapshot == GRADED and knowledge.calls == 0


def test_graded_card_for_another_species_is_not_published():
    jobs = InMemoryJobRepository()
    jobs.create("p1", "p1", "species_mujair")  # graded before a correction
    jobs.update("p1", status="completed", final_card=GRADED)

    class Fallback:
        calls = 0

        def get_for_prediction(self, prediction_id):
            Fallback.calls += 1
            raise RuntimeError("generation down")

    lot = _publish(jobs, Fallback())
    assert Fallback.calls == 1 and lot.knowledge_snapshot is None
