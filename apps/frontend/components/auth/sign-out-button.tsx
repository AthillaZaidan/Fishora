'use client'

import { useRouter } from 'next/navigation'
import { useState } from 'react'
import { SignOut } from '@phosphor-icons/react/dist/ssr'
import { Button } from '@/components/common/button'
import { logout } from '@/lib/api/auth'

/**
 * Signs out and lands on the sign-in screen.
 *
 * The session cookie is httponly, so only the API can clear it: if that call
 * fails the person is still signed in, and saying nothing would leave them
 * believing a shared phone is safe to hand over.
 */
export function SignOutButton({
  compact = false,
  block = false,
}: {
  /** Icon only, for the header bar. */
  compact?: boolean
  block?: boolean
}) {
  const router = useRouter()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function signOut() {
    setBusy(true)
    setError('')
    try {
      await logout()
      router.replace('/account')
      // Every server-rendered page read the old session; drop them.
      router.refresh()
    } catch {
      setError('Could not sign out. Check your connection and try again.')
    } finally {
      setBusy(false)
    }
  }

  if (compact) {
    // No room for a message line in the bar: the icon turns red, the tooltip
    // carries the text, and the live region reads it out.
    return (
      <>
        <button
          type="button"
          onClick={() => void signOut()}
          disabled={busy}
          aria-label="Sign out"
          title={error || 'Sign out'}
          className={[
            'flex min-h-11 min-w-11 items-center justify-center rounded-full',
            'hover:bg-bg-sunken disabled:opacity-45',
            error ? 'text-state-error' : 'text-ink-muted hover:text-ink',
          ].join(' ')}
        >
          <SignOut size={20} aria-hidden />
        </button>
        <span className="sr-only" role="alert">
          {error}
        </span>
      </>
    )
  }

  return (
    <div className="flex flex-col gap-2">
      <Button
        type="button"
        variant="secondary"
        block={block}
        loading={busy}
        icon={<SignOut size={16} aria-hidden />}
        onClick={() => void signOut()}
      >
        Sign out
      </Button>
      {error && (
        <p className="text-body-sm text-state-error" role="alert">
          {error}
        </p>
      )}
    </div>
  )
}
