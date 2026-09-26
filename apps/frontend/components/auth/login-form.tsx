'use client'

import { useRouter } from 'next/navigation'
import { useId, useState } from 'react'
import { Button } from '@/components/common/button'
import { Field } from '@/components/common/field'
import {
  MIN_PASSWORD_LENGTH,
  USERNAME_PATTERN,
  homeFor,
  login,
  register,
  type Role,
  type Session,
} from '@/lib/api/auth'
import { ApiError } from '@/lib/api/errors'

type Mode = 'signin' | 'register'

type FieldErrors = Partial<Record<'displayName' | 'username' | 'password' | 'role', string>>

/**
 * The seeded demo accounts (services/session.py DEMO_USERS). They are public on
 * purpose: someone trying the demo should not have to register first. They sit
 * behind a toggle so the real form stays the way in.
 */
const DEMO_ACCOUNTS = [
  {
    username: 'rian',
    password: 'demo',
    name: 'Rian Setiawan',
    role: 'Operator',
    blurb: 'Identify catches and publish lots.',
  },
  {
    username: 'dewi',
    password: 'demo',
    name: 'Dewi Anggraini',
    role: 'Buyer',
    blurb: 'Browse auctions, place bids, write reviews.',
  },
] as const

const ROLE_OPTIONS: { value: Role; label: string; blurb: string }[] = [
  {
    value: 'operator',
    label: 'Fisher / landing-site operator',
    blurb: 'Photograph catches, confirm the species and put lots up for auction.',
  },
  {
    value: 'buyer',
    label: 'Business buyer',
    blurb: 'Find lots that suit your business and bid on them.',
  },
]

/** Only a same-site path: `next` comes from the URL, so it must not send anyone off-site. */
function safeNext(next: string | undefined): string | null {
  return next && next.startsWith('/') && !next.startsWith('//') ? next : null
}

/**
 * Written here rather than taken from ApiError.userMessage: that table knows
 * the fish and lot endpoints, and would call a wrong password a server error.
 */
function describe(cause: unknown): string {
  if (cause instanceof ApiError) {
    if (cause.status === 401) return 'Wrong username or password.'
    if (cause.status === 422) return 'Some details are not valid. Check the fields and try again.'
    if (cause.kind === 'offline') return 'No connection. Check your network and try again.'
    if (cause.kind === 'timeout') return 'The server took too long to answer. Try again.'
  }
  return 'Something went wrong. Try again.'
}

function validate(
  mode: Mode,
  fields: { displayName: string; username: string; password: string; role: Role | '' }
): FieldErrors {
  const errors: FieldErrors = {}
  if (mode === 'signin') {
    if (!fields.username) errors.username = 'Enter your username.'
    if (!fields.password) errors.password = 'Enter your password.'
    return errors
  }
  if (!fields.displayName) errors.displayName = 'Enter the name buyers and sellers will see.'
  if (!USERNAME_PATTERN.test(fields.username)) {
    errors.username = '3 to 32 characters: lowercase letters, numbers and underscores.'
  }
  if (fields.password.length < MIN_PASSWORD_LENGTH) {
    errors.password = `Use at least ${MIN_PASSWORD_LENGTH} characters.`
  }
  if (!fields.role) errors.role = 'Choose how you will use Fishora.'
  return errors
}

export function LoginForm({
  next,
}: {
  /** Where to go after signing in, e.g. back to the lot a guest wanted to bid on. */
  next?: string
}) {
  const router = useRouter()
  const groupId = useId()
  const [mode, setMode] = useState<Mode>('signin')
  const [displayName, setDisplayName] = useState('')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [role, setRole] = useState<Role | ''>('')
  const [showPassword, setShowPassword] = useState(false)
  const [useDemo, setUseDemo] = useState(false)
  // Which action is running: 'form', or the username of a demo account.
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({})

  function switchMode(to: Mode) {
    setMode(to)
    setError('')
    setFieldErrors({})
  }

  function finish(session: Session) {
    router.replace(safeNext(next) ?? homeFor(session.role))
    // The shell and every page read the session on the server, so they only
    // see the new one once the tree is refetched.
    router.refresh()
  }

  async function run(action: string, attempt: () => Promise<Session>) {
    setBusy(action)
    setError('')
    try {
      finish(await attempt())
      // busy stays set: the page is navigating away, and a second click would
      // only send the same request again.
    } catch (cause) {
      if (cause instanceof ApiError && cause.status === 409) {
        setFieldErrors({ username: 'That username is taken. Choose another one.' })
      } else {
        setError(describe(cause))
      }
      setBusy('')
    }
  }

  function submit() {
    const fields = {
      displayName: displayName.trim(),
      username: username.trim().toLowerCase(),
      password,
      role,
    }
    const errors = validate(mode, fields)
    setFieldErrors(errors)
    if (Object.keys(errors).length > 0) return
    void run('form', () =>
      mode === 'signin'
        ? login(fields.username, fields.password)
        : register({
            username: fields.username,
            password: fields.password,
            display_name: fields.displayName,
            role: fields.role as Role,
          })
    )
  }

  const registering = mode === 'register'

  return (
    <div className="mt-6 flex max-w-md flex-col gap-8">
      <div className="flex rounded-full border border-line p-1" role="group" aria-label="Account">
        {(
          [
            ['signin', 'Sign in'],
            ['register', 'Create account'],
          ] as const
        ).map(([value, label]) => (
          <button
            key={value}
            type="button"
            aria-pressed={mode === value}
            onClick={() => switchMode(value)}
            className={[
              'text-body-sm min-h-11 flex-1 rounded-full px-4 font-medium transition-colors duration-150',
              mode === value ? 'bg-bg-sunken text-ink' : 'text-ink-muted hover:text-ink',
            ].join(' ')}
          >
            {label}
          </button>
        ))}
      </div>

      <form
        className="flex flex-col gap-2"
        noValidate
        onSubmit={(event) => {
          event.preventDefault()
          submit()
        }}
      >
        {registering && (
          <Field
            label="Your name"
            autoComplete="name"
            maxLength={80}
            value={displayName}
            onChange={(event) => setDisplayName(event.target.value)}
            error={fieldErrors.displayName}
          />
        )}
        <Field
          label="Username"
          autoComplete="username"
          autoCapitalize="none"
          autoCorrect="off"
          spellCheck={false}
          maxLength={32}
          value={username}
          onChange={(event) => setUsername(event.target.value)}
          helper={registering ? 'Lowercase letters, numbers and underscores.' : undefined}
          error={fieldErrors.username}
        />
        <Field
          label="Password"
          type={showPassword ? 'text' : 'password'}
          autoComplete={registering ? 'new-password' : 'current-password'}
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          helper={registering ? `At least ${MIN_PASSWORD_LENGTH} characters.` : undefined}
          error={fieldErrors.password}
        />
        <label className="text-body-sm -mt-2 flex min-h-11 items-center gap-3 text-ink-muted">
          <input
            type="checkbox"
            className="size-5"
            checked={showPassword}
            onChange={(event) => setShowPassword(event.target.checked)}
          />
          Show password
        </label>

        {registering && (
          <fieldset className="mt-2 flex flex-col gap-2">
            <legend className="text-label text-ink">I am a</legend>
            {ROLE_OPTIONS.map((option) => (
              <label
                key={option.value}
                className={[
                  'mt-2 flex cursor-pointer items-start gap-3 rounded-[var(--radius-input)] border px-3 py-3',
                  'has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-focus',
                  role === option.value ? 'border-accent' : 'border-line-input',
                ].join(' ')}
              >
                <input
                  type="radio"
                  className="mt-0.5 size-5 shrink-0"
                  name={`${groupId}-role`}
                  value={option.value}
                  checked={role === option.value}
                  onChange={() => setRole(option.value)}
                />
                <span className="flex flex-col gap-0.5">
                  <span className="text-body-sm font-medium text-ink">{option.label}</span>
                  <span className="text-body-sm text-ink-muted">{option.blurb}</span>
                </span>
              </label>
            ))}
            {fieldErrors.role && (
              <p className="text-body-sm text-state-error" role="alert">
                {fieldErrors.role}
              </p>
            )}
          </fieldset>
        )}

        {error && (
          <p className="text-body-sm mt-2 text-state-error" role="alert">
            {error}
          </p>
        )}

        <Button type="submit" block loading={busy === 'form'} disabled={busy !== ''} className="mt-4">
          {registering ? 'Create account' : 'Sign in'}
        </Button>
      </form>

      <section className="flex flex-col gap-3 border-t border-line pt-6">
        <label className="text-body-sm flex min-h-11 items-center gap-3 text-ink">
          <input
            type="checkbox"
            className="size-5"
            checked={useDemo}
            onChange={(event) => setUseDemo(event.target.checked)}
          />
          Use a demo account
        </label>
        {useDemo && (
          <>
            <p className="text-body-sm text-ink-muted">
              Public accounts for trying Fishora, password <span className="font-mono">demo</span>.
              Anyone can sign in to them, so keep real details out.
            </p>
            {DEMO_ACCOUNTS.map((account) => (
              <div key={account.username} className="flex flex-col gap-1">
                <Button
                  type="button"
                  variant="secondary"
                  block
                  loading={busy === account.username}
                  disabled={busy !== ''}
                  onClick={() =>
                    void run(account.username, () => login(account.username, account.password))
                  }
                >
                  Sign in as {account.name} ({account.role})
                </Button>
                <p className="text-body-sm text-center text-ink-muted">{account.blurb}</p>
              </div>
            ))}
          </>
        )}
      </section>
    </div>
  )
}
