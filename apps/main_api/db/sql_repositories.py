from typing import Callable, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.main_api.contracts import KnowledgeJobRecord, PredictionRecord, SpeciesRecord
from apps.main_api.db.models import FishSpecies, KnowledgeJob, Prediction
from apps.main_api.errors import PredictionNotFound


class SqlSpeciesRepository:
    def __init__(self, session_factory: Callable[[], Session]):
        self._session_factory = session_factory

    def get_by_normalized_label(self, label: str) -> SpeciesRecord | None:
        with self._session_factory() as session:
            row = session.scalar(select(FishSpecies).where(FishSpecies.normalized_label == label))
            return self._to_record(row)

    def get_by_id(self, species_id: str) -> SpeciesRecord | None:
        with self._session_factory() as session:
            return self._to_record(session.get(FishSpecies, species_id))

    def list_all(self) -> list[SpeciesRecord]:
        # Ordered by label: an unordered result reshuffles the operator's
        # species picker between requests.
        with self._session_factory() as session:
            rows = session.scalars(
                select(FishSpecies).order_by(FishSpecies.normalized_label)
            ).all()
            return [self._to_record(row) for row in rows]

    @staticmethod
    def _to_record(row: FishSpecies | None) -> SpeciesRecord | None:
        if row is None:
            return None
        return SpeciesRecord(
            id=row.id,
            normalized_label=row.normalized_label,
            common_name_id=row.common_name_id,
            scientific_name=row.scientific_name,
            taxonomic_rank=row.taxonomic_rank,
            taxonomy_status=row.taxonomy_status,
            notes=row.notes,
        )


class SqlPredictionRepository:
    def __init__(self, session_factory: Callable[[], Session]):
        self._session_factory = session_factory

    def create(
        self,
        prediction_id: str,
        image_reference: str,
        predicted_species_id: str,
        confidence: float,
        top_candidates: list[dict[str, object]],
        model_version: str,
    ) -> PredictionRecord:
        row = Prediction(
            id=prediction_id,
            image_reference=image_reference,
            predicted_species_id=predicted_species_id,
            confidence=confidence,
            top_candidates=top_candidates,
            model_version=model_version,
            verification_status="pending",
        )
        with self._session_factory() as session:
            session.add(row)
            session.commit()
        return self._to_record(row)

    def get(self, prediction_id: str) -> PredictionRecord | None:
        with self._session_factory() as session:
            return self._to_record(session.get(Prediction, prediction_id))

    def verify(
        self,
        prediction_id: str,
        verified_species_id: str,
        verification_status: Literal["confirmed", "corrected"],
    ) -> PredictionRecord:
        with self._session_factory() as session:
            row = session.get(Prediction, prediction_id)
            if row is None:
                raise PredictionNotFound(prediction_id)
            row.verification_status = verification_status
            row.verified_species_id = verified_species_id
            session.commit()
        return self._to_record(row)

    @staticmethod
    def _to_record(row: Prediction | None) -> PredictionRecord | None:
        if row is None:
            return None
        return PredictionRecord(
            id=row.id,
            image_reference=row.image_reference,
            predicted_species_id=row.predicted_species_id,
            confidence=row.confidence,
            top_candidates=row.top_candidates,
            model_version=row.model_version,
            verification_status=row.verification_status,
            verified_species_id=row.verified_species_id,
        )


class SqlKnowledgeJobRepository:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def create(self, job_id: str, prediction_id: str, species_id: str):
        row = KnowledgeJob(id=job_id, prediction_id=prediction_id, species_id=species_id, status="processing")
        with self._session_factory() as session:
            existing = session.get(KnowledgeJob, job_id)
            if existing is not None:
                existing.prediction_id = prediction_id
                existing.species_id = species_id
                existing.status = "processing"
                existing.expert_outputs = None
                existing.critic_feedback = None
                existing.final_card = None
                existing.error = None
                existing.trace = None
                existing.completed_at = None
                session.commit()
                return self._to_record(existing)
            session.add(row)
            session.commit()
        return self._to_record(row)

    def get(self, job_id: str):
        with self._session_factory() as session:
            return self._to_record(session.get(KnowledgeJob, job_id))

    def update(self, job_id: str, status: str | None = None, **fields):
        from datetime import datetime, timezone

        with self._session_factory() as session:
            row = session.get(KnowledgeJob, job_id)
            if row is None:
                return None
            if status is not None:
                row.status = status
                if status in ("completed", "failed"):
                    row.completed_at = datetime.now(timezone.utc)
            for k, v in fields.items():
                setattr(row, k, v)
            session.commit()
            return self._to_record(row)

    def list_by_prediction(self, prediction_id: str):
        from sqlalchemy import select

        with self._session_factory() as session:
            rows = session.scalars(select(KnowledgeJob).where(KnowledgeJob.prediction_id == prediction_id)).all()
            return [self._to_record(r) for r in rows]

    @staticmethod
    def _to_record(row: KnowledgeJob | None):
        if row is None:
            return None
        return KnowledgeJobRecord(
            id=row.id,
            prediction_id=row.prediction_id,
            species_id=row.species_id,
            status=row.status,
            expert_outputs=row.expert_outputs,
            critic_feedback=row.critic_feedback,
            final_card=row.final_card,
            error=row.error,
            trace=row.trace,
        )