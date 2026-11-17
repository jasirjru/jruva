/** Small fetch helpers shared by the Plugins, Projects and Scheduled pages. */

export const UNAUTHORIZED = 401;

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

const OFFLINE_MESSAGE = 'Jruva could not be reached. Check your connection and try again.';

function fallbackMessage(status: number): string {
  if (status === UNAUTHORIZED) return 'Sign in to use this.';
  if (status === 404) return 'This item no longer exists. Refresh the page.';
  if (status >= 500) return 'Jruva had a problem on its side. Try again in a moment.';
  return 'Something went wrong. Try again.';
}

async function readDetail(res: Response): Promise<string | null> {
  try {
    const body = await res.json();
    return typeof body?.detail === 'string' ? body.detail : null;
  } catch {
    return null;
  }
}

/** Fetch JSON and turn network failures and non-2xx responses into ApiError with a readable message. */
export async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch {
    throw new ApiError(0, OFFLINE_MESSAGE);
  }
  if (!res.ok) {
    const detail = res.status === UNAUTHORIZED ? null : await readDetail(res);
    throw new ApiError(res.status, detail ?? fallbackMessage(res.status));
  }
  return (await res.json()) as T;
}

export function jsonBody(method: string, body: unknown): RequestInit {
  return { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
}

export function errorMessage(err: unknown): string {
  return err instanceof Error && err.message ? err.message : 'Something went wrong. Try again.';
}

export function isSignedOutError(err: unknown): boolean {
  return err instanceof ApiError && err.status === UNAUTHORIZED;
}

const DATE_FORMAT: Intl.DateTimeFormatOptions = { day: 'numeric', month: 'short', year: 'numeric' };
const DATE_TIME_FORMAT: Intl.DateTimeFormatOptions = { ...DATE_FORMAT, hour: 'numeric', minute: '2-digit' };

export function formatDate(value?: string | null, withTime = false): string | null {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleString(undefined, withTime ? DATE_TIME_FORMAT : DATE_FORMAT);
}
