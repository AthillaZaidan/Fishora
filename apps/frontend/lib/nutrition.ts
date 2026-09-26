import { oneDecimalPlace } from '@/lib/format'
import type { SpeciesLabel } from '@/lib/species'

/**
 * Protein per 100 g edible portion, raw, from published food-composition tables.
 *
 * Every value was read from the cited source on 2026-09-26; none is estimated.
 * `match` says how closely the source item matches the label: a genus-level
 * value is a different species of the same genus. A label with no sourced value
 * is absent, and the card says so rather than borrowing a neighbour's number.
 *
 * TKPI items have no URL of their own: panganku.org serves them from a POST to
 * /id-ID/view with `haha=<code>`. The ASEAN 2000 rows are read from FAO/INFOODS
 * uFiSh1.0, sheet "10 Reftbl_RefDatasets", which reproduces that table.
 *
 * `item` keeps the source's item code and gives the item in English; a local
 * market name stays in brackets where the source item is listed under it.
 */
export interface ProteinFact {
  gramsPer100g: number
  source: string
  item: string
  url: string
  match: 'species' | 'genus'
}

const TKPI = 'TKPI 2017, Kemenkes RI'
const TKPI_URL = 'https://www.panganku.org/id-ID/semua_nutrisi'
const ASEAN = 'ASEAN Food Composition Tables 2000 (via FAO/INFOODS uFiSh1.0)'
const UFISH_URL = 'https://www.fao.org/fileadmin/templates/food_composition/documents/uFiSh1.0.xlsx'
const MYFCD = 'Malaysian Food Composition Database 1997, KKM'
const myfcd = (code: string) =>
  `https://myfcd.moh.gov.my/myfcd97/index.php/site/detail_product/${code}/0/10/-1/0/0/`

// Not found in TKPI, ASEAN, MyFCD, Thai FCD or uFiSh: kuniran (no goatfish item)
// and gembolo (a vernacular name the taxonomy marks as ambiguous).
export const PROTEIN: Partial<Record<SpeciesLabel, ProteinFact>> = {
  bandeng: { gramsPer100g: 20.0, source: TKPI, item: 'GR007 Milkfish (bandeng), fresh', url: TKPI_URL, match: 'species' },
  gelama_bunga: { gramsPer100g: 19.1, source: ASEAN, item: 'AAG101 Croaker, plain, raw', url: UFISH_URL, match: 'species' },
  gulamah: { gramsPer100g: 18.7, source: MYFCD, item: '110048 Soldier croaker (Johnius soldado)', url: myfcd('110048'), match: 'genus' },
  kembung: { gramsPer100g: 21.3, source: TKPI, item: 'GR050 Indian mackerel (oci, kembung), fresh', url: TKPI_URL, match: 'species' },
  mujair: { gramsPer100g: 18.7, source: TKPI, item: 'GR048 Mozambique tilapia (mujahir), fresh', url: TKPI_URL, match: 'species' },
  nila: {
    gramsPer100g: 19.36,
    source: 'Thai Food Composition Database, INMU Mahidol',
    item: 'G98 Nile tilapia, raw',
    url: 'https://inmu.mahidol.ac.th/thaifcd/foodsearch/food_name_result?mode=food_group_result&food_group_id=&food_name=x&selected_id=1112&selected_dbcode=STD&status=A&page_no=1',
    match: 'species',
  },
  senangin: { gramsPer100g: 21.7, source: MYFCD, item: '110100 Threadfin (Senangin)', url: myfcd('110100'), match: 'species' },
  tenggiri: { gramsPer100g: 21.5, source: MYFCD, item: '110052 Narrow-barred Spanish mackerel (tenggiri batang)', url: myfcd('110052'), match: 'species' },
  tuna: { gramsPer100g: 23.6, source: ASEAN, item: 'AAG161 Tuna, yellow-fin, raw', url: UFISH_URL, match: 'genus' },
}

export function proteinFor(label: string): ProteinFact | null {
  return PROTEIN[label as SpeciesLabel] ?? null
}

/** "20 g" or "19.4 g": English decimal point, one decimal at most. */
export function formatProtein(fact: ProteinFact): string {
  return `${oneDecimalPlace(fact.gramsPer100g)} g`
}
