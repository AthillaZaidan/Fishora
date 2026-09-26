'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Fish, SquaresFour, User, Sliders } from '@phosphor-icons/react/dist/ssr'
import { Logo } from '@/components/common/logo'
import { ThemeToggle } from '@/components/common/theme-toggle'
import { Z } from '@/lib/z'

export interface Session {
  id: string
  role: string
  name: string
  username: string
}

const OPERATOR_TABS = [
  { href: '/operator', label: 'Identify', icon: Fish },
  { href: '/operator/lots', label: 'My lots', icon: SquaresFour },
  { href: '/account', label: 'Account', icon: User },
]

const BUYER_TABS = [
  { href: '/marketplace', label: 'Marketplace', icon: SquaresFour },
  { href: '/preferences', label: 'Preferences', icon: Sliders },
  { href: '/account', label: 'Account', icon: User },
]

/** Signed out: the marketplace is public, everything else needs an account. */
const GUEST_TABS = [
  { href: '/marketplace', label: 'Marketplace', icon: SquaresFour },
  { href: '/account', label: 'Sign in', icon: User },
]

const ROLE_LABEL: Record<string, string> = {
  operator: 'Operator',
  buyer: 'Buyer',
}

/** Clears the fixed phone tab bar, which otherwise sits over the last rows of every page. */
const PAGE = 'mx-auto w-full max-w-[1200px] px-4 pb-28 pt-6 lg:px-8 lg:pb-16'

/**
 * The capture screen is a viewport-height column with its own bottom bar, so
 * the shell's vertical padding would push it past the fold and give the page a
 * scrollbar it should never have.
 */
const FLUSH = 'mx-auto w-full max-w-[1200px] px-4 lg:px-8'
const FLUSH_ROUTES = ['/operator']

export function AppShell({
  children,
  session,
}: {
  children: React.ReactNode
  session?: Session | null
}) {
  const pathname = usePathname()
  // Role comes from the session, never from the URL: the pathname says which
  // page you opened, not who you are signed in as.
  const tabs = session
    ? session.role === 'operator'
      ? OPERATOR_TABS
      : BUYER_TABS
    : GUEST_TABS

  return (
    <>
      <header
        className="sticky top-0 border-b border-line bg-surface pt-[env(safe-area-inset-top)]"
        style={{ zIndex: Z.nav }}
      >
        <div className="mx-auto flex h-14 w-full max-w-[1200px] items-center justify-between gap-4 px-4 lg:px-8">
          {/* The wordmark is the way back to the marketing site. */}
          <Link href="/" className="text-h3 flex min-h-11 items-center text-ink">
            <Logo />
          </Link>

          {/* The tab bar below is phone-only, so without this the app has no
              navigation at all above 1024px. */}
          <nav className="hidden items-center gap-1 lg:flex" aria-label="Primary">
            {tabs.map((tab) => {
              const active = pathname === tab.href.split('?')[0]
              return (
                <Link
                  key={tab.href}
                  href={tab.href}
                  aria-current={active ? 'page' : undefined}
                  className={[
                    'text-body-sm flex min-h-11 items-center rounded-full px-3',
                    active ? 'text-ink' : 'text-ink-muted hover:text-ink',
                  ].join(' ')}
                >
                  {tab.label}
                </Link>
              )
            })}
          </nav>

          <div className="flex items-center gap-3">
            {session ? (
              <p className="text-body-sm text-ink-muted">
                <span className="text-ink">{session.name}</span>
                <span className="hidden sm:inline">
                  {' '}
                  · {ROLE_LABEL[session.role] ?? session.role}
                </span>
              </p>
            ) : (
              <Link href="/account" className="text-body-sm flex min-h-11 items-center px-2 text-ink-muted">
                Masuk
              </Link>
            )}
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className={FLUSH_ROUTES.includes(pathname) ? FLUSH : PAGE}>{children}</main>

      <nav
        className="fixed inset-x-0 bottom-0 flex border-t border-line bg-surface pb-[env(safe-area-inset-bottom)] lg:hidden"
        style={{ zIndex: Z.nav }}
        aria-label="Primary, phone"
      >
        {tabs.map((tab) => {
          const Icon = tab.icon
          const active = pathname === tab.href.split('?')[0]
          return (
            <Link
              key={tab.href}
              href={tab.href}
              aria-current={active ? 'page' : undefined}
              className={[
                'flex min-h-14 flex-1 flex-col items-center justify-center gap-1 text-body-sm',
                active ? 'text-ink' : 'text-ink-muted',
              ].join(' ')}
            >
              <Icon size={20} aria-hidden />
              {tab.label}
            </Link>
          )
        })}
      </nav>
    </>
  )
}
