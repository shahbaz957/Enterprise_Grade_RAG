import { NextRequest, NextResponse } from "next/server";

const DEFAULT_BACKEND = "http://127.0.0.1:8000";

function backendUrl(): string {
  return (process.env.RAG_BACKEND_URL || DEFAULT_BACKEND).replace(/\/$/, "");
}

export async function POST(req: NextRequest) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ detail: "Invalid JSON body" }, { status: 400 });
  }

  if (
    !body ||
    typeof body !== "object" ||
    typeof (body as { question?: unknown }).question !== "string" ||
    !(body as { question: string }).question.trim()
  ) {
    return NextResponse.json(
      { detail: "question is required" },
      { status: 400 },
    );
  }

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  const apiKey = process.env.RAG_API_KEY?.trim();
  if (apiKey) {
    headers.Authorization = `Bearer ${apiKey}`;
  }

  const upstream = `${backendUrl()}/query`;
  let res: Response;
  try {
    res = await fetch(upstream, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      cache: "no-store",
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Upstream unreachable";
    return NextResponse.json(
      { detail: `Backend unreachable at ${upstream}: ${message}` },
      { status: 502 },
    );
  }

  const text = await res.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text || res.statusText };
  }

  return NextResponse.json(data, { status: res.status });
}
