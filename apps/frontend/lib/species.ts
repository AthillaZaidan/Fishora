/**
 * The 11 labels the CV model emits. Nothing outside this set is identifiable.
 *
 * Every one has catch photography in `public/`, which is what lets the whole
 * set be offered: a label without a photograph falls back to open water, and a
 * card claiming a named fish while showing ambient sea tells a buyer nothing.
 */
export const SUPPORTED_LABELS = [
  'bandeng',
  'gelama_bunga',
  'gembolo',
  'gulamah',
  'kembung',
  'kuniran',
  'mujair',
  'nila',
  'senangin',
  'tenggiri',
  'tuna',
] as const

export type SpeciesLabel = (typeof SUPPORTED_LABELS)[number]

export interface SpeciesNames {
  /** The English name, or the local name where the fish has no English one. */
  commonName: string
  /**
   * The Indonesian market name the catch is landed and sold under. A proper
   * noun of the market, shown beside the English name, never instead of it.
   * Equal to `commonName` when there is no English name.
   */
  localName: string
  scientificName: string | null
  /** Catch photography for this species, served from /public. */
  photo: string
}

/**
 * The photograph file for each label, as it is actually named in `public/`.
 *
 * Spelled out rather than derived from the label: the extensions differ (both
 * are JPEG, only the suffix varies) and `gelama_bunga` is stored under a
 * shortened stem. Deriving the path would silently miss those and fall back to
 * open water on a card claiming a named fish.
 */
const PHOTOS: Record<SpeciesLabel, string> = {
  bandeng: '/bandeng.jpg',
  gelama_bunga: '/gelama.jpg',
  gembolo: '/gembolo.jpeg',
  gulamah: '/gulamah.jpg',
  kembung: '/kembung.jpg',
  kuniran: '/kuniran.jpg',
  mujair: '/mujair.jpg',
  nila: '/nila.jpg',
  senangin: '/senangin.jpg',
  tenggiri: '/tenggiri.jpeg',
  tuna: '/tuna.jpeg',
}

// Follows the taxonomy seed: gembolo has no scientific name, tuna is genus-level.
// English names are the FishBase common names of the scientific name (tenggiri
// shortened from "Narrow-barred Spanish mackerel", as the trade says). Gembolo
// keeps its local name: it names different fish in different regions, so no
// single English name is true of it.
const NAMES: Record<SpeciesLabel, Omit<SpeciesNames, 'photo'>> = {
  bandeng: { commonName: 'Milkfish', localName: 'Bandeng', scientificName: 'Chanos chanos' },
  gelama_bunga: { commonName: 'Greyfin croaker', localName: 'Gelama Bunga', scientificName: 'Pennahia anea' },
  gembolo: { commonName: 'Gembolo', localName: 'Gembolo', scientificName: null },
  gulamah: { commonName: "Belanger's croaker", localName: 'Gulamah', scientificName: 'Johnius belangerii' },
  kembung: { commonName: 'Indian mackerel', localName: 'Kembung', scientificName: 'Rastrelliger kanagurta' },
  kuniran: { commonName: 'Sulphur goatfish', localName: 'Kuniran', scientificName: 'Upeneus sulphureus' },
  mujair: { commonName: 'Mozambique tilapia', localName: 'Mujair', scientificName: 'Oreochromis mossambicus' },
  nila: { commonName: 'Nile tilapia', localName: 'Nila', scientificName: 'Oreochromis niloticus' },
  senangin: { commonName: 'Fourfinger threadfin', localName: 'Senangin', scientificName: 'Eleutheronema tetradactylum' },
  tenggiri: { commonName: 'Spanish mackerel', localName: 'Tenggiri', scientificName: 'Scomberomorus commerson' },
  tuna: { commonName: 'Tuna', localName: 'Tuna', scientificName: 'Thunnus spp.' },
}

export const SPECIES: Record<SpeciesLabel, SpeciesNames> = Object.fromEntries(
  SUPPORTED_LABELS.map((label) => [label, { ...NAMES[label], photo: PHOTOS[label] }])
) as Record<SpeciesLabel, SpeciesNames>

export function resolveSpecies(label: string): SpeciesNames {
  if (isSpeciesLabel(label)) return SPECIES[label]
  // Open water, not a fish: showing some other species' photograph next to an
  // unrecognised label would misinform a buyer about what is in the lot.
  return { commonName: label, localName: label, scientificName: null, photo: '/sea.jpg' }
}

/** `Milkfish (Bandeng)`, or just `Gembolo` where there is no English name. */
export function displayName(label: string): string {
  const { commonName, localName } = resolveSpecies(label)
  return localName === commonName ? commonName : `${commonName} (${localName})`
}

export function isSpeciesLabel(label: string): label is SpeciesLabel {
  return (SUPPORTED_LABELS as readonly string[]).includes(label)
}
