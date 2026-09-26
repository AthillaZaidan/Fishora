// Used only when nothing better is known: `next dev` on its default port.
const DEV_ORIGIN = 'http://localhost:3000'

/**
 * The origin printed codes point at.
 *
 * NEXT_PUBLIC_SITE_URL wins when set, because a printed card outlives any one
 * deployment and must name the public site. Without it, the browser's own
 * origin is right by definition; on the server the caller passes the request's
 * origin. A hard-coded localhost here once put unusable codes on real cards.
 */
export function siteOrigin(serverFallback?: string): string {
  const configured = process.env.NEXT_PUBLIC_SITE_URL
  if (configured) return configured.replace(/\/+$/, '')
  if (typeof window !== 'undefined') return window.location.origin
  return serverFallback ?? DEV_ORIGIN
}

export function discoverUrl(slug: string, origin: string = siteOrigin()) {
  return `${origin}/discover/${encodeURIComponent(slug)}`
}

/** Where the second code on the printed card points. No store URL: there is no
 *  published app listing, so this resolves to the site. */
export function appUrl(origin: string = siteOrigin()) {
  return process.env.NEXT_PUBLIC_APP_URL?.replace(/\/+$/, '') ?? origin
}

/** `app` is not a lot slug, so the route reserves it as a payload keyword. */
export const APP_QR_KEYWORD = 'app'

/** The shape of a lot's public slug (`<label>-<8 hex>`); the route encodes nothing else. */
export const PUBLIC_SLUG = /^[a-z0-9_-]{1,160}$/
