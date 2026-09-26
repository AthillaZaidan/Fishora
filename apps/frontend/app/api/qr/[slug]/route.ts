import QRCode from 'qrcode'
import { APP_QR_KEYWORD, PUBLIC_SLUG, appUrl, discoverUrl, siteOrigin } from '@/lib/qr'

/** A PNG QR code for a lot's public page, keyed by its public slug (not its id). */
export async function GET(
  request: Request,
  { params }: { params: Promise<{ slug: string }> }
) {
  const { slug } = await params
  // Only a slug-shaped value becomes a code: this route must not turn into a
  // generator for arbitrary URLs under Fishora's name.
  if (slug !== APP_QR_KEYWORD && !PUBLIC_SLUG.test(slug)) {
    return new Response('not found', { status: 404 })
  }
  const origin = siteOrigin(new URL(request.url).origin)
  // The printed card carries two codes: this fish, and Fishora itself.
  const payload = slug === APP_QR_KEYWORD ? appUrl(origin) : discoverUrl(slug, origin)
  const png = await QRCode.toBuffer(payload, { type: 'png', width: 320, margin: 1 })
  return new Response(new Uint8Array(png), { headers: { 'content-type': 'image/png' } })
}
