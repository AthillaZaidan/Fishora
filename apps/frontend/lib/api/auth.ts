import { apiFetch } from './client'

export type Role = 'operator' | 'buyer'

export interface Session {
  id: string
  role: string
  name: string
  username: string
}

/** Mirrors RegisterRequest in apps/main_api/api/auth.py, so the form can say
 * what is wrong before the round trip. */
export const USERNAME_PATTERN = /^[a-z0-9_]{3,32}$/
export const MIN_PASSWORD_LENGTH = 8

const JSON_HEADERS = { 'content-type': 'application/json' }

/** 401 on a wrong username or password. Sets the session cookie. */
export function login(username: string, password: string) {
  return apiFetch<Session>('/api/v1/auth/login', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({ username, password }),
  })
}

/** 409 when the username is taken, 422 when a field fails validation. Signs the new account in. */
export function register(payload: {
  username: string
  password: string
  display_name: string
  role: Role
}) {
  return apiFetch<Session>('/api/v1/auth/register', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  })
}

export function logout() {
  return apiFetch<{ ok: boolean }>('/api/v1/auth/logout', { method: 'POST' })
}

export function getMe() {
  return apiFetch<Session>('/api/v1/auth/me')
}

/** Where someone lands after signing in when nothing asked for a page. */
export function homeFor(role: string): string {
  return role === 'operator' ? '/operator' : '/marketplace'
}
