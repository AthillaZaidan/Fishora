import Image from 'next/image'
import { resolveSpecies, isSpeciesLabel } from '@/lib/species'

/**
 * The catch photograph for a lot.
 *
 * A species we hold photography for shows that species. Anything else shows
 * open water, because putting some other fish next to an unrecognised label
 * misinforms a buyer deciding what to bid on, and ambient water claims nothing.
 *
 * The name is printed over the image either way, so the card is never relying
 * on the photograph alone to say what the lot is.
 *
 * The caller owns the box. className lands on the root, so a caller can pass
 * `absolute inset-0` or `aspect-[4/3] w-full` and either works; the root never
 * sets its own position, because a position utility here would beat the
 * caller's by stylesheet order and silently collapse the box to zero height.
 */
export function SpeciesArt({
  label,
  className = '',
  sizes = '(max-width: 640px) 100vw, 33vw',
  priority = false,
  showName = true,
}: {
  label: string
  className?: string
  sizes?: string
  /** Set on the first card in a grid: it is the LCP element. */
  priority?: boolean
  /** Clear only where the name is already stated next to the image. */
  showName?: boolean
}) {
  const species = resolveSpecies(label)
  const known = isSpeciesLabel(label)

  return (
    <div
      className={`overflow-hidden bg-abyss-900 ${className}`}
      role="img"
      aria-label={known ? `Photo of ${species.commonName}` : `Illustration for ${species.commonName}`}
    >
      {/* Containing block for the layers, kept off the root so the caller's
          position utility is free to win. */}
      <div className="relative size-full">
        <Image
          src={species.photo}
          alt=""
          fill
          sizes={sizes}
          priority={priority}
          className="object-cover"
        />

        {/* The name sits over photographs of very different brightness, so it
            gets its own scrim rather than trusting any one image to be dark
            enough behind it. */}
        {showName && (
          <>
            <div className="absolute inset-x-0 bottom-0 h-1/2 bg-gradient-to-t from-abyss-950/90 to-transparent" />
            <p className="text-num-sm absolute bottom-3 left-3 text-abyss-50">
              {species.commonName}
            </p>
          </>
        )}
      </div>
    </div>
  )
}
