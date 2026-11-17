/** Maps internal provider failures to product language. Does not rewrite stored chat text. */

export function presentUserFacingError(message: string): string {
  const text = (message || "").trim();
  if (!text) return "Jruva is temporarily unavailable. Please try again.";

  const quotaOrAccess = /daily|quota|limit reached|too many requests|sign in|unauthorized|permission|unsupported|not supported|timed out|timeout|incomplete/i.test(text);
  const namesProvider = /gemini|groq|openrouter|fallback engine|selected tier|provider/i.test(text);

  if (quotaOrAccess) {
    return text
      .replace(/gemini|groq|openrouter/gi, "Jruva")
      .replace(/selected tier/gi, "Jruva");
  }
  if (namesProvider) {
    return "Jruva is temporarily unavailable. Please try again.";
  }
  return text;
}

export function presentStatusLabel(message: string): string {
  return (message || "")
    .replace(/gemini|groq|openrouter/gi, "Jruva")
    .replace(/fallback engine/gi, "Jruva");
}
