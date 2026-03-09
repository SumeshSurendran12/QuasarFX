import type { JsonParseResult } from "./types";

export function parseJson<T>(text: string): JsonParseResult<T> {
  const raw = text.trim();
  if (!raw) {
    return { data: null, error: null };
  }

  try {
    return { data: JSON.parse(raw) as T, error: null };
  } catch (error) {
    if (error instanceof Error) {
      return { data: null, error: error.message };
    }
    return { data: null, error: "Invalid JSON format." };
  }
}
