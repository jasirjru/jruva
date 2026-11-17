// Allowlist for image URLs that come from model output (Markdown `![alt](url)`).
// Anything not on the list is never loaded, opened or fetched, so a crafted
// answer cannot beacon data to a third-party host or smuggle a script URL.

const SAFE_DATA_IMAGE = /^data:image\/(png|jpeg|gif|webp);base64,[a-z0-9+/=\s]+$/i;

/** True for a same-origin path such as "/api/v1/image/x.png" (not "//host" or "/\\host"). */
export function isSameOriginPath(url: string): boolean {
  return url.startsWith('/') && !url.startsWith('//') && !url.startsWith('/\\');
}

/** Host of the configured API base URL, or null when the app talks to its own origin. */
function configuredApiHost(apiBaseUrl: string | undefined): string | null {
  if (!apiBaseUrl) return null;
  try {
    const parsed = new URL(apiBaseUrl);
    return parsed.protocol === 'https:' ? parsed.host : null;
  } catch {
    return null;
  }
}

/**
 * Returns true when an image URL may be rendered or downloaded.
 * Allowed: same-origin paths, blob:, base64 PNG/JPEG/GIF/WEBP data URLs, and
 * https URLs on the app's own origin or on the configured API host.
 */
export function isSafeImageUrl(
  rawUrl: string,
  apiBaseUrl: string | undefined = process.env.NEXT_PUBLIC_API_URL,
  appOrigin: string | undefined = typeof window !== 'undefined' ? window.location.origin : undefined,
): boolean {
  if (!rawUrl) return false;
  const url = rawUrl.trim().replace(/[\x00-\x1f\x7f]/g, '');
  if (!url || url.includes('\\')) return false;
  if (isSameOriginPath(url)) return true;
  if (url.toLowerCase().startsWith('blob:')) return true;
  if (url.toLowerCase().startsWith('data:')) return SAFE_DATA_IMAGE.test(url);

  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return false;
  }
  if (parsed.protocol !== 'https:' || parsed.username || parsed.password) return false;

  const allowedHosts = new Set<string>();
  const apiHost = configuredApiHost(apiBaseUrl);
  if (apiHost) allowedHosts.add(apiHost);
  if (appOrigin) {
    try {
      const origin = new URL(appOrigin);
      if (origin.protocol === 'https:') allowedHosts.add(origin.host);
    } catch {
      // Ignore an unparsable origin; only the API host stays allowed.
    }
  }
  return allowedHosts.has(parsed.host);
}
