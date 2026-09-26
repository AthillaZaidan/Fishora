import { AccountSummary } from '@/components/auth/account-summary'
import { LoginForm } from '@/components/auth/login-form'
import { WonLots } from '@/components/buyer/won-lots'
import type { Session } from '@/lib/api/auth'
import { getMeAsServer } from '@/lib/api/server'

export default async function AccountPage({
  searchParams,
}: {
  searchParams: Promise<{ next?: string }>
}) {
  const { next } = await searchParams
  // Read the cookie session here, so someone already signed in sees their
  // account instead of being asked to sign in again every time they come back.
  let session: Session | null = null
  try {
    session = await getMeAsServer()
  } catch {
    // Signed out, or the API is down. The page renders the sign-in form.
  }

  if (!session) {
    return (
      // Centred as a card: signing in is the only thing on this page.
      <div className="mx-auto flex w-full max-w-md flex-col py-6 sm:py-12">
        <h1 className="text-h1 text-center text-ink">Welcome to Fishora</h1>
        <p className="text-body mt-2 text-center text-ink-muted">
          Sign in to identify catches and run auctions, or to bid on fresh lots for your business.
        </p>
        <LoginForm next={next} />
      </div>
    )
  }

  return (
    <>
      <h1 className="text-h1 text-ink">Account</h1>
      <AccountSummary session={session} />
      {/* Won lots load themselves from the buyer-scoped endpoint. */}
      {session.role === 'buyer' && <WonLots />}
    </>
  )
}
