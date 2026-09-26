import type { LandingPoint } from '@/lib/api/commerce'

/**
 * The landing points the backend seeds (apps/main_api/services/landing_points.py).
 * The live list comes from GET /api/v1/landing-points; this copy is only the
 * fallback when that request fails, and the source of readable names offline.
 */
export const FALLBACK_LANDING_POINTS: LandingPoint[] = [
  { id: 'lp_muara_angke', name: 'PPI Muara Angke', latitude: -6.104, longitude: 106.792 },
  { id: 'lp_cilacap', name: 'TPI Cilacap', latitude: -7.732, longitude: 109.015 },
  { id: 'lp_karangsong', name: 'PPI Karangsong', latitude: -6.305, longitude: 108.32 },
]

const NAME_BY_ID: Record<string, string> = Object.fromEntries(
  FALLBACK_LANDING_POINTS.map((point) => [point.id, point.name])
)

/**
 * A readable name for a landing point id.
 *
 * Falls back to the id rather than an empty string: a printed card showing
 * `lp_muara_angke` is poor, but one showing nothing where the origin should be
 * is worse.
 */
export function landingPointName(id: string): string {
  return NAME_BY_ID[id] ?? id
}
