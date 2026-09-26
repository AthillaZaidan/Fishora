import type { Metadata, Viewport } from 'next'
import { AppShell, type Session } from '@/components/app/app-shell'
import { getMeAsServer } from '@/lib/api/server'
import { RootHtml } from '../root-html'

export const metadata: Metadata = {
  title: 'Fishora',
  description: 'Identify your catch and publish auction lots.',
}

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  viewportFit: 'cover',
}

export default async function AppLayout({ children }: { children: React.ReactNode }) {
  // Read the session here rather than guessing the role from the pathname: a
  // buyer visiting an operator URL was shown the operator chrome, and the chip
  // in the bar said whatever the URL said rather than who was signed in.
  let session: Session | null = null
  try {
    session = await getMeAsServer()
  } catch {
    // Signed out, or the API is down. The shell renders its signed-out state.
  }

  return (
    <RootHtml lang="en">
      <AppShell session={session}>{children}</AppShell>
    </RootHtml>
  )
}
