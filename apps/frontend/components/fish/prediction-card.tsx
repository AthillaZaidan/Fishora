'use client'

import { useMemo, useState } from 'react'
import { Button } from '@/components/common/button'
import { TickMeter } from '@/components/common/tick-meter'
import type { IdentificationResult, SpeciesCandidate } from '@/lib/api/fish'
import { confidenceBand, percent } from '@/lib/format'
import { resolveSpecies } from '@/lib/species'

const NOT_LISTED = '__not_listed__'

const VERDICT: Record<ReturnType<typeof confidenceBand>, string> = {
  high: 'High',
  medium: 'Medium',
  low: 'Needs checking',
}

// Low confidence keeps Confirm disabled until the operator picks a candidate.
export interface PredictionCardProps {
  result: IdentificationResult
  onConfirm: (speciesId: string) => void
  onSpeciesNotListed?: () => void
}

export function PredictionCard({
  result,
  onConfirm,
  onSpeciesNotListed,
}: PredictionCardProps) {
  const low = result.status === 'low_confidence_human_verification_required'
  const band = confidenceBand(result.prediction.confidence)
  const sorted = useMemo(
    () => [...result.top_candidates].sort((a, b) => b.confidence - a.confidence),
    [result.top_candidates]
  )
  const [selected, setSelected] = useState<string | null>(low ? null : result.prediction.species_id)

  const canConfirm = selected !== null && selected !== NOT_LISTED

  return (
    <section className="flex flex-col gap-5">
      <div className="flex items-end justify-between gap-4">
        <div>
          <p className="text-num-xl text-ink">{percent(result.prediction.confidence)}</p>
          <p
            className={[
              'text-body-sm mt-1',
              band === 'high' ? 'text-ink' : 'text-state-warn',
            ].join(' ')}
          >
            {VERDICT[band]}
          </p>
        </div>
        <TickMeter value={result.prediction.confidence} />
      </div>

      {low ? (
        <fieldset className="flex flex-col gap-2">
          <legend className="text-label text-ink">Choose the species</legend>
          {sorted.map((candidate) => (
            <CandidateRow
              key={candidate.species_id}
              candidate={candidate}
              checked={selected === candidate.species_id}
              onChange={() => setSelected(candidate.species_id)}
            />
          ))}
          <label className="flex min-h-12 cursor-pointer items-center gap-3 rounded-[var(--radius-input)] border border-line px-3">
            <input
              type="radio"
              name="species"
              value={NOT_LISTED}
              checked={selected === NOT_LISTED}
              onChange={() => {
                setSelected(NOT_LISTED)
                onSpeciesNotListed?.()
              }}
              className="size-4 accent-accent"
            />
            <span className="text-body text-ink">Species not listed</span>
          </label>
        </fieldset>
      ) : (
        <p className="text-h3 text-ink">
          {resolveSpecies(result.prediction.normalized_label).commonName}
        </p>
      )}

      <Button
        size="lg"
        block
        disabled={!canConfirm}
        onClick={() => {
          if (selected && selected !== NOT_LISTED) onConfirm(selected)
        }}
      >
        Confirm
      </Button>
      {selected === NOT_LISTED && (
        // Not a dead end: the operator names the fish from the full list, or
        // goes back and retakes the photo.
        <Button size="lg" variant="secondary" block onClick={() => onSpeciesNotListed?.()}>
          Choose from all species
        </Button>
      )}
    </section>
  )
}

function CandidateRow({
  candidate,
  checked,
  onChange,
}: {
  candidate: SpeciesCandidate
  checked: boolean
  onChange: () => void
}) {
  const name = resolveSpecies(candidate.normalized_label).commonName
  return (
    <label className="flex min-h-12 cursor-pointer items-center justify-between gap-3 rounded-[var(--radius-input)] border border-line px-3">
      <span className="flex items-center gap-3">
        <input
          type="radio"
          name="species"
          value={candidate.species_id}
          checked={checked}
          onChange={onChange}
          className="size-4 accent-accent"
        />
        <span className="text-body text-ink">{name}</span>
      </span>
      <span className="text-num-sm text-ink-muted">{percent(candidate.confidence)}</span>
    </label>
  )
}
