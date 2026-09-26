import Image from 'next/image'

/**
 * The eight steps of the core flow, each with the sentence that step is for.
 *
 * They used to share one sentence repeated eight times, which told a reader
 * nothing and read as unfinished.
 */
const STEPS = [
  { verb: 'Catch', line: 'A boat lands, and the catch is grouped by species and size.', crop: 8 },
  { verb: 'Identify', line: 'One photo returns a ranked species prediction with its confidence.', crop: 22 },
  { verb: 'Verify', line: 'The operator confirms or corrects it. Nothing publishes unverified.', crop: 36 },
  { verb: 'Explain', line: 'Retrieval builds a knowledge card from verified sources only.', crop: 50 },
  { verb: 'Publish', line: 'Volume, size, opening price, and landing point become a live lot.', crop: 64 },
  { verb: 'Match', line: 'Buyers see lots that fit their stated use, price, and radius.', crop: 78 },
  { verb: 'Bid', line: 'Bidding runs by the kilogram until the auction closes.', crop: 90 },
  { verb: 'Fishora QR', line: 'The winner prints a card that carries the fish knowledge forward.', crop: 15 },
]

/**
 * A continuously advancing strip of the eight steps.
 *
 * Auto-advancing rather than a scroll container: the eight steps are breadth
 * the reader should see exists, not a decision they have to make, and a strip
 * that never moves reads as a static list with a scrollbar bolted on. The
 * animation is a `transform` on a duplicated track, so it composites on the GPU
 * and never touches layout or the scroll position.
 *
 * It pauses on hover and on keyboard focus, because a line of text that slides
 * away while you are reading it is hostile.
 */
export function FlowStrip() {
  return (
    <div
      id="flow"
      role="group"
      aria-label="How Fishora works"
      // overflow-hidden, not overflow-x-auto: no scrollbar, and the track is
      // moved by the animation rather than by the scroll position.
      className="marquee group relative overflow-hidden"
    >
      <div className="marquee-track flex w-max gap-4 px-4 group-hover:[animation-play-state:paused] group-focus-within:[animation-play-state:paused]">
        {STEPS.map((step) => (
          <FlowPanel key={step.verb} step={step} />
        ))}
        {/* A second pass so the loop has somewhere to go. Hidden from assistive
            tech, which should hear the eight steps once. */}
        {STEPS.map((step) => (
          <FlowPanel key={`echo-${step.verb}`} step={step} aria-hidden />
        ))}
      </div>

      {/* The strip runs to both edges, so it is faded rather than cut. */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-y-0 left-0 w-12 bg-gradient-to-r from-abyss-950 to-transparent"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-y-0 right-0 w-12 bg-gradient-to-l from-abyss-950 to-transparent"
      />
    </div>
  )
}

function FlowPanel({
  step,
  'aria-hidden': hidden,
}: {
  step: (typeof STEPS)[number]
  'aria-hidden'?: boolean
}) {
  return (
    <article
      aria-hidden={hidden}
      // Narrow enough that several steps are visible at once. At 86vw only one
      // panel and a sliver of the next fit, which made the section look empty.
      className="w-[78vw] shrink-0 overflow-hidden rounded-2xl border border-abyss-800 sm:w-[20rem]"
    >
      {/* Not SpeciesArt: these are steps, not species, and SpeciesArt prints
          its own label, which repeated the verb below it. Each step takes a
          different slice of the same water so the panels are not identical. */}
      <div className="relative aspect-[16/9] w-full overflow-hidden bg-abyss-900" aria-hidden>
        <Image
          src="/sea.jpg"
          alt=""
          fill
          sizes="(max-width: 640px) 78vw, 20rem"
          className="object-cover opacity-80"
          style={{ objectPosition: `${step.crop}% 50%` }}
        />
      </div>
      <div className="p-5">
        <h3 className="text-h3">{step.verb}</h3>
        <p className="text-body-sm mt-2 text-abyss-200">{step.line}</p>
      </div>
    </article>
  )
}
