'use server'

import { cookies } from 'next/headers'
import { ApiError, type ApiFetchOptions } from '@/lib/api/client'
import type { ActionResult } from '@/lib/api/action-result'
import {
  declareSpeciesManually,
  getKnowledge,
  identifyFish,
  verifySpecies,
  type IdentificationResult,
  type KnowledgeResult,
  type ManualEntryResult,
} from '@/lib/api/fish'

// A server action runs on the Next server, whose fetch carries no browser
// cookies. The identify routes require an operator session, so it is forwarded.
async function withSession(): Promise<ApiFetchOptions> {
  const jar = await cookies()
  const cookie = jar
    .getAll()
    .map((entry) => `${entry.name}=${entry.value}`)
    .join('; ')
  return cookie ? { headers: { cookie } } : {}
}

function fail(error: unknown): ActionResult<never> {
  if (error instanceof ApiError) {
    return {
      ok: false,
      kind: error.kind,
      userMessage:
        error.status === 401
          ? 'Your session has expired. Sign in again as a fisher to continue.'
          : error.status === 403
            ? 'This step needs a fisher (operator) account.'
            : error.userMessage,
      retryable: error.retryable,
      status: error.status,
    }
  }
  throw error
}

export async function identifyCatch(
  formData: FormData
): Promise<ActionResult<IdentificationResult>> {
  const file = formData.get('file')
  if (!(file instanceof File)) {
    return {
      ok: false,
      kind: 'image_invalid',
      userMessage: 'Unsupported image format. Use JPG or PNG.',
      retryable: false,
      status: 400,
    }
  }
  try {
    return { ok: true, data: await identifyFish(file, undefined, await withSession()) }
  } catch (error) {
    return fail(error)
  }
}

export async function confirmSpecies(
  predictionId: string,
  verifiedSpeciesId: string
): Promise<ActionResult<{
  prediction_id: string
  predicted_species_id: string
  verified_species_id: string
  verification_status: 'confirmed' | 'corrected'
}>> {
  try {
    return { ok: true, data: await verifySpecies(predictionId, verifiedSpeciesId, await withSession()) }
  } catch (error) {
    return fail(error)
  }
}

export async function loadKnowledge(
  predictionId: string
): Promise<ActionResult<KnowledgeResult>> {
  try {
    return { ok: true, data: await getKnowledge(predictionId, await withSession()) }
  } catch (error) {
    return fail(error)
  }
}

export async function declareSpecies(
  formData: FormData,
  speciesId: string
): Promise<ActionResult<ManualEntryResult>> {
  const file = formData.get('file')
  if (!(file instanceof File)) {
    return {
      ok: false,
      kind: 'image_invalid',
      userMessage: 'Unsupported image format. Use JPG or PNG.',
      retryable: false,
      status: 400,
    }
  }
  try {
    return { ok: true, data: await declareSpeciesManually(file, speciesId, await withSession()) }
  } catch (error) {
    return fail(error)
  }
}
