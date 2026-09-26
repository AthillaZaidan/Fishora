'use client'

import { useRouter } from 'next/navigation'
import { useState } from 'react'
import { Button } from '@/components/common/button'
import { Select } from '@/components/common/select'
import { login, logout } from '@/lib/api/commerce'
import { ApiError } from '@/lib/api/errors'

type Session = { id: string; role: string; name: string; username: string }

/**
 * The seeded demo accounts, from apps/main_api/services/session.py.
 *
 * A free-text username and password asked people to know credentials that only
 * exist in a source file, so the sign-in screen was a guessing game. These are
 * demo accounts on a demo deployment; listing them is the point.
 */
const DEMO_ACCOUNTS = [
  {
    username: 'rian',
    password: 'demo',
    name: 'Rian Setiawan',
    role: 'Operator',
    blurb: 'Identifikasi tangkapan dan terbitkan lot.',
  },
  {
    username: 'dewi',
    password: 'demo',
    name: 'Dewi Anggraini',
    role: 'Pembeli',
    blurb: 'Telusuri lelang, ajukan penawaran, tulis ulasan.',
  },
] as const

/** Only a same-site path: `next` comes from the URL, so it must not send anyone off-site. */
function safeNext(next: string | undefined): string | null {
  return next && next.startsWith('/') && !next.startsWith('//') ? next : null
}

export function LoginForm({
  initialSession = null,
  next,
}: {
  initialSession?: Session | null
  /** Where to go after signing in, e.g. back to the lot a guest wanted to bid on. */
  next?: string
}) {
  const router = useRouter()
  const [username, setUsername] = useState<string>(DEMO_ACCOUNTS[0].username)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [session, setSession] = useState<Session | null>(initialSession)

  const chosen = DEMO_ACCOUNTS.find((account) => account.username === username) ?? DEMO_ACCOUNTS[0]

  async function submit() {
    setBusy(true)
    setError('')
    try {
      setSession(await login(chosen.username, chosen.password))
      const target = safeNext(next)
      if (target) {
        router.replace(target)
        return
      }
      // The shell reads the session on the server, so it only picks up the new
      // role once the tree is refetched.
      router.refresh()
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.userMessage : 'Gagal masuk.')
    } finally {
      setBusy(false)
    }
  }

  if (session) {
    return (
      <div className="mt-6 flex flex-col gap-3">
        <p className="text-h3 text-ink">{session.name}</p>
        <p className="text-body-sm text-ink-muted">
          Masuk sebagai {session.role === 'operator' ? 'operator' : 'pembeli'}.
        </p>
        <Button
          type="button"
          variant="secondary"
          onClick={async () => {
            await logout()
            setSession(null)
            router.refresh()
          }}
        >
          Keluar
        </Button>
      </div>
    )
  }

  return (
    <form
      className="mt-6 flex flex-col gap-4"
      onSubmit={(event) => {
        event.preventDefault()
        void submit()
      }}
    >
      <Select
        label="Akun demo"
        value={username}
        onChange={(event) => setUsername(event.target.value)}
        helper={chosen.blurb}
        options={DEMO_ACCOUNTS.map((account) => ({
          value: account.username,
          label: `${account.name} (${account.role})`,
        }))}
      />

      {error && (
        <p className="text-body-sm text-state-error" role="alert">
          {error}
        </p>
      )}

      <Button type="submit" block loading={busy}>
        Masuk sebagai {chosen.name}
      </Button>
    </form>
  )
}
