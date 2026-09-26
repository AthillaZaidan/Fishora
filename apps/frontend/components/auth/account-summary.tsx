import { SignOutButton } from '@/components/auth/sign-out-button'
import type { Session } from '@/lib/api/auth'

const ROLE_LABEL: Record<string, string> = {
  operator: 'Fisher / landing-site operator',
  buyer: 'Business buyer',
}

export function AccountSummary({ session }: { session: Session }) {
  return (
    <section className="mt-6 flex max-w-md flex-col gap-6">
      <dl className="flex flex-col gap-4 rounded-[var(--radius-card)] border border-line bg-surface p-4">
        <div className="flex flex-col gap-1">
          <dt className="text-label text-ink-muted">Name</dt>
          <dd className="text-h3 text-ink">{session.name}</dd>
        </div>
        <div className="flex flex-col gap-1">
          <dt className="text-label text-ink-muted">Role</dt>
          <dd className="text-body text-ink">{ROLE_LABEL[session.role] ?? session.role}</dd>
        </div>
        <div className="flex flex-col gap-1">
          <dt className="text-label text-ink-muted">Username</dt>
          <dd className="text-body font-mono text-ink">{session.username}</dd>
        </div>
      </dl>
      <SignOutButton block />
    </section>
  )
}
