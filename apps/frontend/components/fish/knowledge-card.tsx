import type { ComponentType } from 'react'
import {
  ArrowsClockwise,
  Barbell,
  CookingPot,
  ForkKnife,
  Info,
  ShieldCheck,
  ShoppingCart,
  Waves,
} from '@phosphor-icons/react/dist/ssr'
import type { IconProps } from '@phosphor-icons/react'
import { TaxonomyQualifier } from '@/components/fish/taxonomy-qualifier'
import { SourceList } from '@/components/fish/source-list'
import type { KnowledgeCard } from '@/lib/api/fish'
import { normaliseDashes } from '@/lib/format'
import { formatProtein, proteinFor } from '@/lib/nutrition'

// The verified surface. ReviewsRatings stays a sibling, never a child.
export interface KnowledgeCardViewProps {
  card: KnowledgeCard
  /** Normalized CV label, so MIXED_TAXONOMY on tuna can name the genus. */
  label: string
}

// The development seed marks its cards with this limitation (in English, or in
// the Indonesian it was first written in); scripts/seed_demo_lots.py.
const SAMPLE_CARD = /sample data|data contoh|fixture/i

type Provenance = 'verified' | 'sample' | 'unavailable'

/**
 * Only a card that cites sources has earned the verified header. A seeded
 * sample card and an identity-only card (generation was down at publication)
 * both have none, and saying "verified" over them would be false.
 */
export function cardProvenance(card: KnowledgeCard): Provenance {
  if (card.sources.length > 0) return 'verified'
  if (card.limitations.some((item) => SAMPLE_CARD.test(item))) return 'sample'
  return 'unavailable'
}

const HEADER: Record<Provenance, string> = {
  verified: 'Verified knowledge',
  sample: 'Sample card',
  unavailable: 'Knowledge not yet available',
}

/**
 * The fish at a glance, one labelled row per fact, as in the Fish Knowledge
 * Card the team uses for buyers. Every row but protein comes from the
 * generated, source-cited card; protein comes from a food-composition table and
 * names that table under its value.
 */
export function KnowledgeCardView({ card, label }: KnowledgeCardViewProps) {
  const protein = proteinFor(label)
  const provenance = cardProvenance(card)
  const HeaderIcon = provenance === 'verified' ? ShieldCheck : Info

  return (
    <article className="rounded-2xl bg-surface px-5 py-5">
      <header
        className={[
          'text-body-sm mb-4 inline-flex items-center gap-1.5',
          provenance === 'verified' ? 'text-verified' : 'text-ink-muted',
        ].join(' ')}
      >
        <HeaderIcon className="size-4" weight={provenance === 'verified' ? 'fill' : 'regular'} aria-hidden />
        {HEADER[provenance]}
      </header>

      <div className="flex flex-col gap-5">
        {card.scientific_name && (
          <p className="text-body max-w-[65ch] text-ink-muted italic">
            {display(card.scientific_name)}
          </p>
        )}

        <TaxonomyQualifier status={card.taxonomy_status} label={label} />

        <dl className="flex flex-col gap-2">
          <FactRow icon={ForkKnife} label="Taste" value={card.taste} />
          <FactRow icon={Waves} label="Texture" value={card.texture} />
          <FactRow icon={CookingPot} label="Suitable for" value={joined(card.processing_methods)} />
          <FactRow icon={ArrowsClockwise} label="Substitutes" value={joined(card.similar_or_substitute_species)} />
          <FactRow icon={ShoppingCart} label="Potential buyers" value={joined(card.potential_buyer_segments)} />
          <FactRow
            icon={Barbell}
            label="Protein per 100 g"
            value={protein ? formatProtein(protein) : null}
            empty="No sourced data yet"
            note={
              protein && (
                <>
                  <a href={protein.url} className="underline underline-offset-2" target="_blank" rel="noreferrer">
                    {protein.source}
                  </a>
                  , {protein.item}
                  {protein.match === 'genus' && ' (another species in the same genus)'}
                </>
              )
            }
          />
        </dl>

        {card.physical_characteristics && (
          <Field heading="Physical traits" body={card.physical_characteristics} />
        )}
        <ChipList heading="Commercial uses" items={card.commercial_uses} />

        {card.limitations.length > 0 && (
          <section aria-labelledby="knowledge-limitations">
            <h3 id="knowledge-limitations" className="text-label mb-2 text-ink">
              Limitations
            </h3>
            <ul className="text-body-sm flex flex-col gap-2 text-ink-muted">
              {card.limitations.map((item) => (
                <li key={item}>{display(item)}</li>
              ))}
            </ul>
          </section>
        )}

        {card.sources.length > 0 && (
          <section aria-labelledby="knowledge-sources">
            <h3 id="knowledge-sources" className="text-label mb-2 text-ink">
              Sources
            </h3>
            <SourceList sources={card.sources} />
          </section>
        )}
      </div>
    </article>
  )
}

function FactRow({
  icon: Icon,
  label,
  value,
  empty = 'No verified evidence yet',
  note,
}: {
  icon: ComponentType<IconProps>
  label: string
  value: string | null
  empty?: string
  note?: React.ReactNode
}) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-line px-3 py-2.5">
      <span className="grid size-9 shrink-0 place-items-center rounded-full bg-bg-sunken text-ink" aria-hidden>
        <Icon size={18} />
      </span>
      <div className="flex min-w-0 flex-1 flex-col gap-0.5 sm:flex-row sm:items-baseline sm:gap-4">
        <dt className="text-label shrink-0 text-ink sm:w-36">{label}</dt>
        <dd className="min-w-0 flex-1">
          <p className={value ? 'text-body text-ink' : 'text-body-sm text-ink-muted'}>
            {value ? display(value) : empty}
          </p>
          {note && <p className="text-body-sm mt-0.5 text-ink-muted">{note}</p>}
        </dd>
      </div>
    </div>
  )
}

function Field({ heading, body }: { heading: string; body: string }) {
  return (
    <section>
      <h3 className="text-label mb-1 text-ink">{heading}</h3>
      <p className="text-body max-w-[65ch] text-ink-muted">{display(body)}</p>
    </section>
  )
}

function ChipList({ heading, items }: { heading: string; items: string[] }) {
  if (items.length === 0) return null
  return (
    <section>
      <h3 className="text-label mb-2 text-ink">{heading}</h3>
      <ul className="text-body-sm flex flex-col gap-1 text-ink-muted">
        {items.map((item) => (
          <li key={item}>{display(item)}</li>
        ))}
      </ul>
    </section>
  )
}

function joined(items: string[]): string | null {
  return items.length ? items.join(', ') : null
}

/** Strips long dashes and leaked markdown headings from generated text. */
function display(text: string): string {
  return normaliseDashes(text.replace(/^#{1,6}\s+/gm, ''))
}
