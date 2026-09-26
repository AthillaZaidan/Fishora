import { LoginForm } from '@/components/auth/login-form'
import { WonLots } from '@/components/buyer/won-lots'
import { listLots, type Lot } from '@/lib/api/commerce'
import { getMeAsServer } from '@/lib/api/server'

export default async function AccountPage({
  searchParams,
}: {
  searchParams: Promise<{ next?: string }>
}) {
  const { next } = await searchParams
  // Read the cookie session here. Without it the form starts from empty local
  // state, so someone already signed in was asked to pick an account again
  // every time they came back to this page.
  let session = null
  try {
    session = await getMeAsServer()
  } catch {
    // Signed out, or the API is down. The form renders its sign-in state.
  }

  // Fetch first, render after: JSX built inside a try/catch is not protected.
  let won: Lot[] | null = null
  if (session?.role === 'buyer') {
    try {
      const allocated = await listLots('status=allocated')
      won = allocated.filter((lot) => lot.allocated_buyer_id === session.id)
    } catch {
      won = []
    }
  }

  return (
    <>
      <h1 className="text-h1 text-ink">Akun</h1>
      <LoginForm initialSession={session} next={next} />
      {won && <WonLots lots={won} />}
    </>
  )
}
