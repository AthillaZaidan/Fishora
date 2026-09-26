'use client'

import Link from 'next/link'
import { useEffect, useMemo, useState } from 'react'
import { Crosshair, Star } from '@phosphor-icons/react/dist/ssr'
import { Button } from '@/components/common/button'
import { Field } from '@/components/common/field'
import { Select } from '@/components/common/select'
import { ApiError } from '@/lib/api/client'
import {
  getRecommendations,
  matchedLotIds,
  previewPreferences,
  savePreferences,
  type BuyerPreferences,
  type PreferenceDraft,
  type PreferencePreview,
} from '@/lib/api/preferences'
import { kilograms, kilometres, rupiahPerKg } from '@/lib/format'
import { Z } from '@/lib/z'

/** PRD 5.2: the buyer segments the MVP prioritises, plus exporters. The API
 *  reads each against the buyer segments on a lot's knowledge card. */
const BUSINESS_TYPES = [
  { value: 'restaurant', label: 'Restaurant' },
  { value: 'hotel', label: 'Hotel' },
  { value: 'catering', label: 'Catering' },
  { value: 'supermarket', label: 'Supermarket' },
  { value: 'seafood_retailer', label: 'Seafood retailer' },
  { value: 'processor', label: 'Processor' },
  { value: 'distributor', label: 'Distributor' },
  { value: 'exporter', label: 'Exporter' },
]

// Worded as the English knowledge cards word them. The API folds inflection
// ("fried", "frying", "Deep-fried") and ignores negated words ("not too oily"),
// so these only need to name the idea.
const USES = ['fried', 'grilled', 'fillet', 'smoked', 'steamed', 'boiled', 'fish balls', 'curry', 'canned']
const CHARS = ['savory', 'firm', 'soft', 'mild', 'sweet', 'oily', 'white flesh']

// Chips the Indonesian form saved. Shown as their English chip so a returning
// buyer's profile opens as they left it.
const LEGACY_CHIPS: Record<string, string> = {
  digoreng: 'fried',
  goreng: 'fried',
  dibakar: 'grilled',
  bakar: 'grilled',
  diasap: 'smoked',
  pindang: 'boiled',
  bakso: 'fish balls',
  gurih: 'savory',
  padat: 'firm',
  lembut: 'soft',
  berminyak: 'oily',
  'daging putih': 'white flesh',
}

interface Place {
  key: string
  label: string
  latitude: number
  longitude: number
}

// The landing points and the cities around them. Matching only reaches lots
// landed within the serviceability radius (100 km), so a buyer has to be able
// to say where they are; one fixed origin in Jakarta hid every Cilacap and
// Karangsong lot from everyone.
const PLACES: Place[] = [
  { key: 'muara_angke', label: 'Jakarta, Muara Angke', latitude: -6.104, longitude: 106.792 },
  { key: 'jakarta', label: 'Jakarta, city centre', latitude: -6.1754, longitude: 106.8272 },
  { key: 'tangerang', label: 'Tangerang', latitude: -6.1783, longitude: 106.6319 },
  { key: 'bekasi', label: 'Bekasi', latitude: -6.2383, longitude: 106.9756 },
  { key: 'bogor', label: 'Bogor', latitude: -6.595, longitude: 106.8166 },
  { key: 'karawang', label: 'Karawang', latitude: -6.3227, longitude: 107.3376 },
  { key: 'bandung', label: 'Bandung', latitude: -6.9175, longitude: 107.6191 },
  { key: 'indramayu', label: 'Indramayu, Karangsong', latitude: -6.305, longitude: 108.32 },
  { key: 'cirebon', label: 'Cirebon', latitude: -6.732, longitude: 108.5523 },
  { key: 'cilacap', label: 'Cilacap', latitude: -7.732, longitude: 109.015 },
  { key: 'purwokerto', label: 'Purwokerto', latitude: -7.4214, longitude: 109.2345 },
  { key: 'tegal', label: 'Tegal', latitude: -6.8694, longitude: 109.1402 },
]

const CURRENT = 'current'

type Location = { key: string; latitude: number; longitude: number }
type GeoState = 'idle' | 'locating' | 'denied' | 'unavailable' | 'unsupported'
type Counts = { matched: number; inRange: number }

const GEO_MESSAGES: Record<Exclude<GeoState, 'idle' | 'locating'>, string> = {
  denied: 'Location access is blocked in this browser. Choose the nearest city from the list instead.',
  unavailable: 'Your location could not be found. Choose the nearest city from the list instead.',
  unsupported: 'This browser cannot share a location. Choose the nearest city from the list instead.',
}

/** Kilometres between two points, for matching a saved location to a listed place. */
function distanceKm(a: { latitude: number; longitude: number }, b: { latitude: number; longitude: number }) {
  const rad = Math.PI / 180
  const dLat = (b.latitude - a.latitude) * rad
  const dLon = (b.longitude - a.longitude) * rad
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(a.latitude * rad) * Math.cos(b.latitude * rad) * Math.sin(dLon / 2) ** 2
  return 12742 * Math.asin(Math.sqrt(h))
}

function locationFrom(saved: BuyerPreferences | null): Location | null {
  if (!saved) return null
  // Within 2 km reads as that place: the old form saved (-6.1, 106.8), which
  // is Muara Angke to the nearest kilometre.
  const place = PLACES.find((candidate) => distanceKm(candidate, saved) <= 2)
  return place
    ? { key: place.key, latitude: place.latitude, longitude: place.longitude }
    : { key: CURRENT, latitude: saved.latitude, longitude: saved.longitude }
}

function chipsFrom(values: string[] | undefined): string[] {
  const chips: string[] = []
  for (const value of values ?? []) {
    const chip = LEGACY_CHIPS[value.trim().toLowerCase()] ?? value.trim()
    if (chip && !chips.includes(chip)) chips.push(chip)
  }
  return chips
}

const grouped = new Intl.NumberFormat('id-ID', { maximumFractionDigits: 2 })

/** `"68000.00"` -> `"68.000"`, the way a buyer would type it back. */
function amountText(value: string | null | undefined): string {
  if (value === null || value === undefined || value === '') return ''
  const number = Number(value)
  return Number.isFinite(number) ? grouped.format(number) : ''
}

/**
 * A typed amount as a number: `68.000`, `68,000` and `68000` are all 68000,
 * and `12,5` is 12.5. Empty is null (no limit); anything else is NaN.
 *
 * Sending the raw text read "68.000" as 68, a limit no lot could meet.
 */
export function parseAmount(raw: string): number | null {
  const text = raw.replace(/\s|rp|kg|\/+/gi, '')
  if (!text) return null
  if (/^\d{1,3}([.,]\d{3})+$/.test(text)) return Number(text.replace(/[.,]/g, ''))
  if (/^\d+$/.test(text)) return Number(text)
  if (/^\d+[.,]\d{1,2}$/.test(text)) return Number(text.replace(',', '.'))
  return Number.NaN
}

function saveErrorMessage(cause: unknown): string {
  if (cause instanceof ApiError) {
    if (cause.status === 401) return 'Your session has ended. Sign in again to save your profile.'
    if (cause.status === 403) return 'Only your own buyer account can save this profile.'
    if (cause.status === 422) return 'Some values were not accepted. Check the price, volume and location.'
    if (cause.kind === 'offline' || cause.kind === 'timeout') return cause.userMessage
  }
  return 'Your profile could not be saved. Try again.'
}

function lots(count: number) {
  return `${count} ${count === 1 ? 'lot' : 'lots'}`
}

export function PreferenceForm({
  buyerId,
  initial,
  initialCounts = null,
  loadFailed = false,
}: {
  buyerId: string
  initial: BuyerPreferences | null
  initialCounts?: Counts | null
  /** The saved profile could not be read; saving now would overwrite it blind. */
  loadFailed?: boolean
}) {
  const [businessType, setBusinessType] = useState(initial?.business_type ?? BUSINESS_TYPES[0].value)
  const [uses, setUses] = useState(() => chipsFrom(initial?.intended_uses))
  const [chars, setChars] = useState(() => chipsFrom(initial?.characteristics))
  const [maxPrice, setMaxPrice] = useState(() => amountText(initial?.max_price_per_kg))
  const [minVolume, setMinVolume] = useState(() => amountText(initial?.min_quantity_kg))
  const [location, setLocation] = useState<Location | null>(() => locationFrom(initial))
  const [geo, setGeo] = useState<GeoState>('idle')
  const [attempted, setAttempted] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [justSaved, setJustSaved] = useState(false)
  const [counts, setCounts] = useState<Counts | null>(initialCounts)
  const [preview, setPreview] = useState<{ key: string; data: PreferencePreview | null } | null>(null)

  const price = parseAmount(maxPrice)
  const volume = parseAmount(minVolume)
  const priceError =
    price !== null && (Number.isNaN(price) || price <= 0)
      ? 'Enter a price such as 68.000, or leave it empty for no limit.'
      : undefined
  const volumeError =
    volume !== null && Number.isNaN(volume)
      ? 'Enter a weight such as 50, or leave it empty for no minimum.'
      : undefined

  const draft: PreferenceDraft | null =
    location && !priceError && !volumeError
      ? {
          business_type: businessType,
          intended_uses: uses,
          characteristics: chars,
          // Null when empty: no box filled is "no limit", and zero would match nothing.
          max_price_per_kg: price === null ? null : String(price),
          min_quantity_kg: volume === null ? null : String(volume),
          latitude: location.latitude,
          longitude: location.longitude,
        }
      : null
  const key = draft ? JSON.stringify(draft) : ''

  // What was last saved, in the same shape, so "unsaved changes" is a comparison
  // and not a flag every input has to remember to clear.
  const [savedKey, setSavedKey] = useState(() => {
    if (!initial) return null
    const location = locationFrom(initial)
    const price = parseAmount(amountText(initial.max_price_per_kg))
    const volume = parseAmount(amountText(initial.min_quantity_kg))
    return JSON.stringify({
      business_type: initial.business_type,
      intended_uses: chipsFrom(initial.intended_uses),
      characteristics: chipsFrom(initial.characteristics),
      max_price_per_kg: price === null ? null : String(price),
      min_quantity_kg: volume === null ? null : String(volume),
      latitude: location?.latitude,
      longitude: location?.longitude,
    } satisfies Record<keyof PreferenceDraft, unknown>)
  })
  const dirty = key !== savedKey

  useEffect(() => {
    if (!key) return
    // Debounced: the numeric fields fire on every keystroke, and one request
    // per character would both hammer the API and race its own responses.
    const controller = new AbortController()
    const timer = setTimeout(() => {
      previewPreferences(buyerId, JSON.parse(key) as PreferenceDraft, controller.signal)
        .then((data) => setPreview({ key, data }))
        .catch(() => {
          // A preview is a convenience. Losing it must not break the form.
          if (!controller.signal.aborted) setPreview({ key, data: null })
        })
    }, 350)
    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [key, buyerId])

  // Only a preview of exactly this draft counts; an older one would show the
  // count for choices the buyer has already changed.
  const current = preview?.key === key ? preview.data : null
  const previewPending = Boolean(key) && preview?.key !== key

  const edited = <T,>(set: (value: T) => void) => (value: T) => {
    setJustSaved(false)
    setError('')
    set(value)
  }

  const toggle = (list: string[], setList: (next: string[]) => void, value: string) =>
    edited(setList)(list.includes(value) ? list.filter((item) => item !== value) : [...list, value])

  const locationOptions = useMemo(() => {
    const options = [
      { value: '', label: 'Choose your location' },
      ...PLACES.map((place) => ({ value: place.key, label: place.label })),
    ]
    if (location?.key === CURRENT) {
      options.push({
        value: CURRENT,
        label: `My location (${location.latitude.toFixed(3)}, ${location.longitude.toFixed(3)})`,
      })
    }
    return options
  }, [location])

  const businessOptions = BUSINESS_TYPES.some((type) => type.value === businessType)
    ? BUSINESS_TYPES
    : [...BUSINESS_TYPES, { value: businessType, label: businessType.replace(/_/g, ' ') }]

  function chooseLocation(value: string) {
    setGeo('idle')
    if (value === CURRENT) return
    const place = PLACES.find((candidate) => candidate.key === value)
    edited(setLocation)(
      place ? { key: place.key, latitude: place.latitude, longitude: place.longitude } : null
    )
  }

  function useMyLocation() {
    if (typeof navigator === 'undefined' || !('geolocation' in navigator)) {
      setGeo('unsupported')
      return
    }
    setGeo('locating')
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setGeo('idle')
        // Rounded to about 100 m: matching works in kilometres, and a
        // street-level fix is more than a buying profile needs to keep.
        edited(setLocation)({
          key: CURRENT,
          latitude: Math.round(position.coords.latitude * 1000) / 1000,
          longitude: Math.round(position.coords.longitude * 1000) / 1000,
        })
      },
      (failure) => setGeo(failure.code === failure.PERMISSION_DENIED ? 'denied' : 'unavailable'),
      { enableHighAccuracy: false, timeout: 10_000, maximumAge: 10 * 60_000 }
    )
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    setAttempted(true)
    if (!draft || loadFailed) return
    setBusy(true)
    setError('')
    try {
      await savePreferences(buyerId, draft)
      setSavedKey(key)
      setJustSaved(true)
      // The count comes from the matching engine after the save, not from the
      // preview: it is the number the marketplace badges will show.
      try {
        const recommendations = await getRecommendations(buyerId)
        setCounts({
          matched: matchedLotIds(recommendations).length,
          inRange: recommendations.items.length,
        })
      } catch {
        setCounts(null)
      }
    } catch (cause) {
      setJustSaved(false)
      setError(saveErrorMessage(cause))
    } finally {
      setBusy(false)
    }
  }

  const nearest = current?.nearest_landing_point ?? null
  const outOfRange = current !== null && nearest !== null && nearest.distance_km > current.radius_km

  return (
    <form className="flex flex-col gap-8 pb-28 lg:pb-0" onSubmit={submit} noValidate>
      {loadFailed && (
        <p role="alert" className="text-body-sm rounded-[var(--radius-card)] border border-state-error p-3 text-state-error">
          Your saved profile could not be loaded, so saving is turned off to avoid replacing it.
          Reload the page to try again.
        </p>
      )}

      <fieldset className="flex flex-col gap-3">
        <legend className="text-h3 text-ink">Where you are</legend>
        <p className="text-body-sm -mt-1 text-ink-muted">
          Only lots landed within delivery range of this location are matched.
        </p>
        <Select
          label="Location"
          value={location?.key ?? ''}
          onChange={(event) => chooseLocation(event.target.value)}
          options={locationOptions}
          error={attempted && !location ? 'Choose a location, or use your current one.' : undefined}
        />
        <div className="flex flex-wrap items-center gap-3">
          <Button
            type="button"
            variant="secondary"
            size="sm"
            icon={<Crosshair size={16} aria-hidden />}
            loading={geo === 'locating'}
            onClick={useMyLocation}
          >
            Use my current location
          </Button>
        </div>
        <div aria-live="polite" className="text-body-sm">
          {geo !== 'idle' && geo !== 'locating' && <p className="text-state-error">{GEO_MESSAGES[geo]}</p>}
          {nearest && !outOfRange && (
            <p className="text-ink-muted">
              Nearest landing point: {nearest.name}, {kilometres(nearest.distance_km)} away.
            </p>
          )}
          {nearest && outOfRange && (
            <p className="text-state-error">
              No landing point is within {kilometres(current.radius_km)}. The nearest, {nearest.name}, is{' '}
              {kilometres(nearest.distance_km)} away, so no lots can be matched from here.
            </p>
          )}
        </div>
      </fieldset>

      <Select
        label="Business type"
        helper="Compared with the buyers each lot's knowledge card names."
        value={businessType}
        onChange={(event) => edited(setBusinessType)(event.target.value)}
        options={businessOptions}
      />

      <ChipSet
        legend="What you cook or sell"
        hint="Compared with the processing methods and uses on each lot's knowledge card."
        options={[...USES, ...uses.filter((use) => !USES.includes(use))]}
        selected={uses}
        onToggle={(value) => toggle(uses, setUses, value)}
      />

      <ChipSet
        legend="Qualities you look for"
        hint="Taste and texture, compared with the knowledge card. A card saying “not oily” does not count as oily."
        options={[...CHARS, ...chars.filter((char) => !CHARS.includes(char))]}
        selected={chars}
        onToggle={(value) => toggle(chars, setChars, value)}
      />

      <fieldset>
        <legend className="text-h3 text-ink">Price and volume limits</legend>
        <p className="text-body-sm mt-1 text-ink-muted">Leave empty for no limit.</p>
        <div className="mt-3 grid gap-4 sm:grid-cols-2">
          <Field
            label="Maximum price per kg"
            inputMode="decimal"
            prefix="Rp"
            placeholder="68.000"
            value={maxPrice}
            error={priceError}
            helper={price ? `Read as ${rupiahPerKg(price)}` : 'No limit'}
            onChange={(event) => edited(setMaxPrice)(event.target.value)}
          />
          <Field
            label="Minimum volume"
            inputMode="decimal"
            suffix="kg"
            placeholder="50"
            value={minVolume}
            error={volumeError}
            helper={volume ? `Read as ${kilograms(volume)}` : 'No minimum'}
            onChange={(event) => edited(setMinVolume)(event.target.value)}
          />
        </div>
      </fieldset>

      <div
        className="fixed inset-x-0 bottom-14 flex items-center justify-between gap-3 border-t border-line bg-surface px-4 py-3 pb-[calc(0.75rem+env(safe-area-inset-bottom))] lg:static lg:bottom-auto lg:border-0 lg:px-0"
        style={{ zIndex: Z.actionBar }}
      >
        <div aria-live="polite" className="min-w-0">
          <Summary
            hasLocation={Boolean(location)}
            dirty={dirty}
            saved={savedKey !== null}
            justSaved={justSaved}
            preview={current}
            previewPending={previewPending}
            counts={counts}
          />
        </div>
        <Button type="submit" loading={busy} disabled={loadFailed || (!dirty && savedKey !== null)}>
          {savedKey === null ? 'Save profile' : 'Save changes'}
        </Button>
      </div>

      {error && (
        <p role="alert" className="text-body-sm text-state-error">
          {error}
        </p>
      )}
    </form>
  )
}

function Summary({
  hasLocation,
  dirty,
  saved,
  justSaved,
  preview,
  previewPending,
  counts,
}: {
  hasLocation: boolean
  dirty: boolean
  saved: boolean
  justSaved: boolean
  preview: PreferencePreview | null
  previewPending: boolean
  counts: Counts | null
}) {
  if (!hasLocation) {
    return <p className="text-body-sm text-ink-muted">Choose a location to see which lots match.</p>
  }

  // Saved and unchanged: the real count from the recommendations endpoint.
  if (saved && !dirty) {
    return (
      <>
        <p className="text-num-sm tabular-nums text-ink">
          {counts ? `${lots(counts.matched)} match your profile` : 'Profile saved'}
        </p>
        <p className="text-body-sm text-ink-muted">
          {justSaved ? 'Saved. ' : ''}
          {counts && counts.matched > 0 ? (
            <Link href="/marketplace?matched=1" className="inline-flex items-center gap-1 text-ink underline underline-offset-2">
              <Star size={12} weight="fill" className="text-accent" aria-hidden />
              See matched lots
            </Link>
          ) : counts ? (
            `${lots(counts.inRange)} in range, none a strong match yet.`
          ) : null}
        </p>
      </>
    )
  }

  // A draft: the same computation run on the unsaved choices.
  return (
    <>
      <p className="text-num-sm tabular-nums text-ink">
        {preview ? `${lots(preview.matched)} would match` : previewPending ? 'Counting lots…' : 'Count unavailable'}
      </p>
      <p className="text-body-sm text-ink-muted">
        {preview ? `${lots(preview.in_range)} in range. ` : ''}
        {saved ? 'Changes not saved yet.' : 'Not saved yet.'}
      </p>
    </>
  )
}

function ChipSet({
  legend,
  hint,
  options,
  selected,
  onToggle,
}: {
  legend: string
  hint: string
  options: string[]
  selected: string[]
  onToggle: (value: string) => void
}) {
  return (
    <fieldset>
      <legend className="text-h3 text-ink">{legend}</legend>
      <p className="text-body-sm mt-1 text-ink-muted">{hint}</p>
      <div className="mt-3 flex flex-wrap gap-2">
        {options.map((option) => (
          <button
            key={option}
            type="button"
            aria-pressed={selected.includes(option)}
            className={`min-h-11 rounded-full border px-4 transition-colors ${
              selected.includes(option)
                ? 'border-ink bg-bg-sunken text-ink'
                : 'border-line text-ink-muted hover:text-ink'
            }`}
            onClick={() => onToggle(option)}
          >
            {option}
          </button>
        ))}
      </div>
    </fieldset>
  )
}
