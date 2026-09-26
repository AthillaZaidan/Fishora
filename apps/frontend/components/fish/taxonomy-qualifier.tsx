import { WarningCircle } from '@phosphor-icons/react/dist/ssr'
import type { TaxonomyStatus } from '@/lib/api/fish'

// Anything other than VERIFIED_TAXONOMY must stay visible beside the name.
export interface TaxonomyQualifierProps {
  status: TaxonomyStatus
  label: string
}

const COPY: Record<Exclude<TaxonomyStatus, 'VERIFIED_TAXONOMY'>, string> = {
  TAXONOMY_REVIEW_REQUIRED:
    'The scientific name needs expert review. This identification is not yet locked to a single species.',
  MEDIUM_CONFIDENCE_LABEL_AMBIGUITY:
    'This label is used for more than one species. Expert confirmation is still required.',
  MIXED_TAXONOMY:
    'Taxonomy is locked at genus level. The exact species has not been determined.',
}

const TUNA_MIXED =
  'Taxonomy is locked at the genus Thunnus spp. until expert verification.'

export function TaxonomyQualifier({ status, label }: TaxonomyQualifierProps) {
  if (status === 'VERIFIED_TAXONOMY') return null

  const message =
    status === 'MIXED_TAXONOMY' && label === 'tuna' ? TUNA_MIXED : COPY[status]

  return (
    <p className="text-body-sm flex items-start gap-1.5 text-state-warn">
      <WarningCircle className="mt-0.5 size-4 shrink-0" weight="fill" aria-hidden />
      <span>{message}</span>
    </p>
  )
}
