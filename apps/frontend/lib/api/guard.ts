import 'server-only'

import { redirect } from 'next/navigation'
import { getMeAsServer } from './server'

export type Session = Awaited<ReturnType<typeof getMeAsServer>>

/**
 * Server-side page guard: returns the signed-in user when they have `role`.
 * Signed out, it sends them to sign in and back to `next`; signed in with the
 * other role, it sends them to their own home. The API enforces the same rule,
 * so this only spares people a page whose every action would fail.
 */
export async function requireRole(role: 'operator' | 'buyer', next: string): Promise<Session> {
  let me: Session | null = null
  try {
    me = await getMeAsServer()
  } catch {
    me = null
  }
  // redirect() throws, so it stays outside the try above.
  if (!me) redirect(`/account?next=${encodeURIComponent(next)}`)
  if (me.role !== role) redirect(me.role === 'operator' ? '/operator' : '/marketplace')
  return me
}
