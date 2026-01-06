/**
 * API utility with authentication header support.
 */

const API_BASE_URL = 'http://localhost:8000'

/**
 * Get the API key from environment variable.
 */
function getApiKey(): string {
  return import.meta.env.VITE_API_KEY || ''
}

/**
 * Create headers with authentication.
 */
function createHeaders(additionalHeaders?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...additionalHeaders,
  }

  const apiKey = getApiKey()
  if (apiKey) {
    headers['X-API-Key'] = apiKey
  }

  return headers
}

/**
 * Make an authenticated API request.
 *
 * @param endpoint - API endpoint (e.g., '/api/conversations')
 * @param options - Fetch options (method, body, etc.)
 * @returns Fetch response
 */
export async function apiFetch(
  endpoint: string,
  options: RequestInit = {}
): Promise<Response> {
  const url = `${API_BASE_URL}${endpoint}`

  const headers = createHeaders(
    options.headers as Record<string, string> | undefined
  )

  return fetch(url, {
    ...options,
    headers,
  })
}

/**
 * GET request with authentication.
 */
export async function apiGet(endpoint: string): Promise<Response> {
  return apiFetch(endpoint, { method: 'GET' })
}

/**
 * POST request with authentication.
 */
export async function apiPost(endpoint: string, body?: unknown): Promise<Response> {
  return apiFetch(endpoint, {
    method: 'POST',
    body: body ? JSON.stringify(body) : undefined,
  })
}

/**
 * DELETE request with authentication.
 */
export async function apiDelete(endpoint: string): Promise<Response> {
  return apiFetch(endpoint, { method: 'DELETE' })
}
