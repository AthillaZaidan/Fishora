import Image from 'next/image'
import Link from 'next/link'
import { IdentificationDemo } from '@/components/marketing/identification-demo'
import type { PredictionCard } from '@/components/fish/prediction-card'
import { LotCard } from '@/components/lot/lot-card'
import { MatchReasons } from '@/components/lot/match-reasons'
import { FlowStrip } from '@/components/marketing/flow-strip'
import { HeroDescent } from '@/components/marketing/hero-descent'
import { LandingNav } from '@/components/marketing/landing-nav'
import { SpeciesArt } from '@/components/fish/species-art'
import { Logo } from '@/components/common/logo'
import type { components } from '@/lib/api/schema'

type Lot = components['schemas']['LotResponse']

const DEMO_LOT: Lot = {
  id: 'lot_tenggiri_1',
  prediction_id: 'pred_ok',
  operator_id: 'op_rian',
  species_id: 'species_tenggiri',
  landing_point_id: 'lp_muara_angke',
  quantity_kg: '24.000',
  size_category: 'L',
  starting_price_per_kg: '68000.00',
  status: 'active',
  auction_starts_at: '2026-08-24T10:00:00+00:00',
  auction_ends_at: '2026-08-24T14:00:00+00:00',
  public_slug: 'tenggiri-lot1',
  allocated_buyer_id: null,
  current_highest_per_kg: '70000.00',
  serviceability_radius_km: 100,
  batch_index: 1,
  batch_size: 1,
}

const DEMO_REASONS = [
  { criterion: 'intended_use', met: true, detail: 'suitable for frying', value: 'frying' },
  { criterion: 'characteristics', met: true, detail: 'matches your preferred characteristics', value: 'savoury' },
  { criterion: 'price', met: true, detail: 'Rp 68,000/kg', value: '68000' },
  { criterion: 'volume', met: true, detail: '24 kg', value: '24 kg' },
  { criterion: 'distance', met: true, detail: '37 km', value: '37 km' },
]

const DEMO_ID: Parameters<typeof PredictionCard>[0]['result'] = {
  prediction_id: 'pred_demo',
  model_version: 'v1',
  status: 'confident_prediction',
  prediction: { species_id: 'species_tenggiri', normalized_label: 'tenggiri', confidence: 0.91 },
  top_candidates: [
    { species_id: 'species_tenggiri', normalized_label: 'tenggiri', confidence: 0.91 },
    { species_id: 'species_kembung', normalized_label: 'kembung', confidence: 0.06 },
  ],
  threshold: 0.8,
  verification_status: 'pending',
}

/** One container for the page. Without it every line sat 16px from the left edge of a 1440px screen. */
const SHELL = 'mx-auto w-full max-w-[1200px] px-4 lg:px-8'

/** DESIGN.md 3.6: landing sections breathe more than app surfaces. */
const SECTION = 'py-20 md:py-28 lg:py-40'

export function LandingPage() {
  return (
    <div className="bg-abyss-950 text-abyss-50">
      <LandingNav />

      <main>
        <HeroDescent />

        <section
          data-block="gap"
          id="platform"
          // Extra head room on phones only: the copy column is the full width
          // there, so without it the heading lands straight over the fisherman's
          // face. At md the photograph sits beside the text and needs none.
          // No overflow-hidden here. An overflow-clipped ancestor becomes the
          // nearest scroll container for `animation-timeline: view()`, and since
          // it never scrolls the reveal animations inside resolve against a dead
          // timeline. The backdrop clips itself instead.
          className={`relative isolate ${SECTION} pt-72 md:pt-28 lg:pt-40`}
        >
          <FishermanBackdrop />
          <div className={`relative ${SHELL}`}>
            <h2 className="reveal text-display-2 max-w-[18ch]">
              Indonesian boats land hundreds of species. Buyers order a handful.
            </h2>
            <dl className="reveal-stagger mt-12 flex flex-col gap-8 md:flex-row md:gap-16">
              <Figure value="100 km" caption="serviceability radius, MVP" />
              <Figure value="5 to 10" caption="species in the MVP model" />
              <Figure value="under 90s" caption="from photo to published lot" />
            </dl>
          </div>
        </section>

        <section data-block="flow" className={SECTION}>
          <p className={`reveal text-eyebrow text-ink-faint ${SHELL}`}>THE FLOW</p>
          <h2 className={`reveal text-h1 mt-2 max-w-[20ch] ${SHELL}`}>
            Eight steps from a boat to a buyer.
          </h2>
          {/* Full bleed on purpose: the strip runs edge to edge past the shell. */}
          <div className="mt-10">
            <FlowStrip />
          </div>
        </section>

        <section data-block="identify" id="operators" className={SECTION}>
          <div className={`${SHELL} lg:grid lg:grid-cols-12 lg:items-center lg:gap-8`}>
            {/* Copy leads in the DOM on both viewports; only the visual order swaps. */}
            <div className="reveal lg:order-2 lg:col-span-5">
              <h2 className="text-h1">The model proposes. A person decides.</h2>
              <p className="text-body mt-3 max-w-[34ch] text-abyss-200">
                Nothing publishes until an operator confirms the species on the landing floor.
              </p>
            </div>
            <div className="reveal mt-8 lg:order-1 lg:col-span-6 lg:mt-0">
              <IdentificationDemo result={DEMO_ID} />
            </div>
          </div>
        </section>

        <section data-block="match" id="buyers" className={SECTION}>
          <div className={SHELL}>
            <p className="reveal text-eyebrow text-ink-faint">WHY IT MATCHED</p>
            <h2 className="reveal text-h1 mt-2">Need, then supply, then the reason.</h2>
            <div className="reveal-stagger mt-10 grid gap-6 lg:grid-cols-2 lg:items-start">
              <LotCard lot={DEMO_LOT} matched />
              <MatchReasons reasons={DEMO_REASONS} />
            </div>
          </div>
        </section>

        <section data-block="knowledge" id="knowledge" className={SECTION}>
          <div className={SHELL}>
            <h2 className="reveal text-h1 max-w-[20ch]">What a shopper can learn from a QR.</h2>

            {/* Exactly five cells for the five things a knowledge card answers.
                A three-column grid left a hole at the end, because five items do
                not fill six cells. */}
            <div className="reveal-stagger mt-10 grid gap-4 md:grid-cols-3">
              <Cell className="md:col-span-2 md:row-span-2">
                <SpeciesArt
                  label="tenggiri"
                  className="aspect-[16/10] w-full"
                  sizes="(max-width: 768px) 100vw, (max-width: 1279px) 62vw, 750px"
                />
                <CellBody title="Taste and texture">
                  Savoury and not especially oily. Firm, with a fine grain.
                </CellBody>
              </Cell>

              {/* Tuna, not kembung: only tenggiri and tuna are 1600px sources.
                  The rest are 520x390, which this cell upscales past 2x on a
                  retina screen and renders soft. */}
              <Cell>
                <SpeciesArt
                  label="tuna"
                  className="aspect-[16/9] w-full"
                  sizes="(max-width: 768px) 100vw, (max-width: 1279px) 31vw, 370px"
                />
                <CellBody title="Cooking">Fresh loins, grilled, steamed.</CellBody>
              </Cell>

              <Cell className="bg-abyss-850">
                <CellBody title="Commercial uses">Fillet. Steak. Restaurants.</CellBody>
              </Cell>

              <Cell className="bg-gradient-to-b from-abyss-900 to-abyss-800">
                <CellBody title="Substitutes">Indian mackerel. Tuna.</CellBody>
              </Cell>

              <Cell className="md:col-span-2">
                <CellBody title="Sources">
                  <span className="text-num-sm">1 verified source</span>
                </CellBody>
              </Cell>
            </div>
          </div>
        </section>

        <section data-block="close" id="access" className={`${SECTION} border-t border-abyss-900`}>
          <div className={SHELL}>
            <h2 className="reveal text-display-2 max-w-[20ch]">
              Every catch already carries its value. Fishora makes it legible.
            </h2>
            <div className="reveal mt-10 flex flex-col gap-3 sm:flex-row">
              <Link
                href="/account"
                className="text-body-sm inline-flex min-h-11 items-center justify-center rounded-full bg-accent px-6 font-medium text-accent-ink"
              >
                Request access
              </Link>
              <Link
                href="/marketplace"
                className="text-body-sm inline-flex min-h-11 items-center justify-center rounded-full border border-line-strong px-6 font-medium text-ink"
              >
                Open the marketplace
              </Link>
            </div>
          </div>
        </section>

      </main>

      <SiteFooter />
    </div>
  )
}

/**
 * The fisherman behind the statement, faded into the abyss.
 *
 * The source is a 447px square, so it is held to the right of the section
 * rather than stretched across a desktop viewport: past roughly 1.6x it starts
 * to look like an upscale instead of a photograph. On phones the viewport is
 * near the source width already, so it runs full width there.
 *
 * The image is masked radially rather than clipped, so it has no straight edge
 * on any side. Scrims then handle contrast: a flat overlay dark enough for
 * `display-2` would bury the photograph, so a vertical fade into the page
 * ground pairs with a horizontal one under the copy column, which keeps the
 * text on near solid colour while the figure stays visible beside it.
 */
function FishermanBackdrop() {
  return (
    <div aria-hidden className="pointer-events-none absolute inset-0 -z-10 overflow-hidden">
      <div className="fade-radial absolute inset-y-0 right-0 w-full md:w-[46rem]">
        <Image
          src="/nelayan.jpeg"
          alt=""
          fill
          // Below a 240vh hero, so never the LCP element.
          sizes="(max-width: 768px) 100vw, 46rem"
          className="object-cover object-[50%_6%]"
        />
      </div>

      {/* Into the page ground at both edges, so the band has no seam. */}
      <div className="absolute inset-0 bg-gradient-to-b from-abyss-950/90 via-abyss-950/20 to-abyss-950" />
      {/* Under the copy column at md and up. */}
      <div className="absolute inset-0 bg-abyss-950/60 md:bg-transparent md:bg-gradient-to-r md:from-abyss-950 md:via-abyss-950/80 md:to-abyss-950/15" />
    </div>
  )
}

function Figure({ value, caption }: { value: string; caption: string }) {
  return (
    <div className="border-t border-abyss-800 pt-4">
      <dt className="text-num-xl text-accent">{value}</dt>
      <dd className="text-body-sm mt-2 text-abyss-200">{caption}</dd>
    </div>
  )
}

function Cell({ className = '', children }: { className?: string; children: React.ReactNode }) {
  return (
    <article className={`flex flex-col overflow-hidden rounded-2xl border border-abyss-800 ${className}`}>
      {children}
    </article>
  )
}

function CellBody({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="flex-1 p-5">
      <p className="text-label text-abyss-200">{title}</p>
      <p className="text-body mt-2 text-abyss-100">{children}</p>
    </div>
  )
}

const FOOTER_PRODUCT = [
  { href: '/marketplace', label: 'Marketplace' },
  { href: '/preferences', label: 'Buyer preferences' },
  { href: '/operator', label: 'Operator app' },
  { href: '/account', label: 'Sign in' },
]

const FOOTER_PAGE = [
  { href: '#platform', label: 'The gap' },
  { href: '#flow', label: 'The flow' },
  { href: '#operators', label: 'Identification' },
  { href: '#knowledge', label: 'Knowledge cards' },
]

function SiteFooter() {
  return (
    <footer className="border-t border-abyss-900 py-16">
      <div className={`${SHELL} grid gap-10 md:grid-cols-3`}>
        <div>
          <Logo className="text-h3" />
          <p className="text-body-sm mt-3 max-w-[32ch] text-abyss-200">
            Species identification, verified knowledge, and a B2B auction for the catch that has just
            landed.
          </p>
        </div>

        <FooterColumn title="Product" links={FOOTER_PRODUCT} internal />
        <FooterColumn title="This page" links={FOOTER_PAGE} />
      </div>

      <div className={`${SHELL} mt-12 flex flex-col gap-2 border-t border-abyss-900 pt-8`}>
        <p className="text-body-sm text-abyss-200">
          <a className="inline-flex min-h-11 items-center" href="mailto:halo@fishora.id">
            halo@fishora.id
          </a>
        </p>
        <p className="text-body-sm text-ink-faint">Jakarta, Indonesia</p>
      </div>
    </footer>
  )
}

function FooterColumn({
  title,
  links,
  internal,
}: {
  title: string
  links: { href: string; label: string }[]
  internal?: boolean
}) {
  return (
    <div>
      <p className="text-label text-abyss-200">{title}</p>
      <ul className="mt-3 flex flex-col">
        {links.map((link) => (
          <li key={link.href}>
            {/* 44px rows: a footer of small links at gap-2 is the most common
                tap-accuracy failure on a marketing site. */}
            {internal ? (
              <Link className="text-body-sm flex min-h-11 items-center text-abyss-100" href={link.href}>
                {link.label}
              </Link>
            ) : (
              <a className="text-body-sm flex min-h-11 items-center text-abyss-100" href={link.href}>
                {link.label}
              </a>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}
