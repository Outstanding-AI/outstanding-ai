"""Coordinate AI case-context reads and public V4 request assembly."""

from __future__ import annotations

from dataclasses import dataclass

from src.api.models.requests import CaseContext

from . import context_evidence as evidence
from .context_projection import assemble_case_context
from .context_reader import ContextHydrationError, ContextReadRepository, LakeReader
from .models import DraftCandidate


@dataclass
class BatchHydrationResult:
    """One candidate's context or controlled hydration error."""

    candidate: DraftCandidate
    context: CaseContext | None = None
    error: ContextHydrationError | None = None


class CaseContextHydrator:
    """Coordinate tenant-scoped reads while preserving the public hydration API."""

    def __init__(
        self,
        tenant_id: str,
        reader: LakeReader,
        *,
        current_source_map: dict[str, str] | None = None,
    ) -> None:
        self.tenant_id = str(tenant_id)
        self.reader = reader
        self.current_source_map = dict(current_source_map or {})
        self._reads = ContextReadRepository(
            self.tenant_id,
            reader,
            current_source_map=self.current_source_map,
        )

    def _source(self, canonical_view: str) -> str:
        """Compatibility seam for current-source-map validation callers."""

        return self._reads.source(canonical_view)

    def hydrate_candidate(self, candidate: DraftCandidate) -> CaseContext:
        """Hydrate one candidate through the same scoped readers as batch mode."""

        party = self._reads.load_party(candidate.party_id)
        lane = self._reads.load_lane(candidate.lane_id)
        lane_ids = candidate.lane_ids()
        obligations_by_lane = self._reads.load_lane_obligations_batch(lane_ids)
        case_ids = [str(candidate.collection_case_id)] if candidate.collection_case_id else []
        case_id = str(candidate.collection_case_id or "")
        return assemble_case_context(
            candidate=candidate,
            party=party,
            lane=lane,
            obligations=evidence.candidate_obligations(candidate, obligations_by_lane, lane_ids),
            party_contacts=self._reads.load_party_contacts(candidate.party_id),
            history=self._reads.load_lane_history(candidate.lane_id),
            actual_sent_scope_history=self._reads.load_actual_sent_scope_history(
                candidate.party_id
            ),
            case_thread=self._reads.load_case_threads_batch(case_ids).get(case_id),
            case_temporal_evidence=self._reads.load_case_temporal_invoice_evidence_batch(
                case_ids
            ).get(case_id, []),
            case_commitment_evidence=self._reads.load_case_commitment_evidence_batch(case_ids).get(
                case_id, []
            ),
        )

    def hydrate_batch(self, candidates: list[DraftCandidate]) -> list[BatchHydrationResult]:
        """Hydrate candidates with one current-projection read per shape.

        Missing party/lane data is isolated to the candidate, preserving the
        established partial-failure behaviour and the original input ordering.
        """

        if not candidates:
            return []

        party_ids = sorted({str(candidate.party_id) for candidate in candidates})
        lane_ids = sorted({lane_id for candidate in candidates for lane_id in candidate.lane_ids()})
        case_ids = sorted(
            {
                str(candidate.collection_case_id)
                for candidate in candidates
                if candidate.collection_case_id
            }
        )

        parties_by_id = self._reads.load_parties_batch(party_ids)
        lanes_by_id = self._reads.load_lanes_batch(lane_ids)
        obligations_by_lane = self._reads.load_lane_obligations_batch(lane_ids)
        contacts_by_party = self._reads.load_party_contacts_batch(party_ids)
        history_by_lane = self._reads.load_lane_history_batch(lane_ids)
        actual_sent_scope_by_party = self._reads.load_actual_sent_scope_history_batch(party_ids)
        case_threads_by_id = self._reads.load_case_threads_batch(case_ids)
        case_temporal_evidence_by_id = self._reads.load_case_temporal_invoice_evidence_batch(
            case_ids
        )
        case_commitment_evidence_by_id = self._reads.load_case_commitment_evidence_batch(case_ids)

        results: list[BatchHydrationResult] = []
        for candidate in candidates:
            party_id = str(candidate.party_id)
            lane_id = str(candidate.lane_id)
            case_id = str(candidate.collection_case_id or "")
            candidate_lane_ids = candidate.lane_ids()
            try:
                party = parties_by_id.get(party_id)
                if party is None:
                    raise ContextHydrationError(f"Party not found in regional Silver: {party_id}")
                lane = lanes_by_id.get(lane_id)
                if lane is None:
                    raise ContextHydrationError(
                        f"Collection lane not found in regional Silver: {lane_id}"
                    )
                context = assemble_case_context(
                    candidate=candidate,
                    party=party,
                    lane=lane,
                    obligations=evidence.candidate_obligations(
                        candidate, obligations_by_lane, candidate_lane_ids
                    ),
                    party_contacts=contacts_by_party.get(party_id, []),
                    history=history_by_lane.get(lane_id, []),
                    actual_sent_scope_history=actual_sent_scope_by_party.get(party_id, []),
                    case_thread=case_threads_by_id.get(case_id),
                    case_temporal_evidence=case_temporal_evidence_by_id.get(case_id, []),
                    case_commitment_evidence=case_commitment_evidence_by_id.get(case_id, []),
                )
            except ContextHydrationError as exc:
                results.append(BatchHydrationResult(candidate=candidate, error=exc))
            else:
                results.append(BatchHydrationResult(candidate=candidate, context=context))
        return results


__all__ = ["BatchHydrationResult", "CaseContextHydrator", "ContextHydrationError"]
