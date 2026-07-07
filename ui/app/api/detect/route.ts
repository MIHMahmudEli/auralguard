import { NextRequest, NextResponse } from "next/server";

// Server-side proxy to the Hugging Face Space so the API URL / token stay private
// and the browser never deals with CORS.
export const runtime = "nodejs";
export const maxDuration = 60;

const API_URL = process.env.AURALGUARD_API_URL ?? "http://localhost:7860";
const API_TOKEN = process.env.AURALGUARD_API_TOKEN; // optional (private Space)

export async function POST(req: NextRequest) {
  try {
    const form = await req.formData();
    const file = form.get("file");
    if (!(file instanceof Blob)) {
      return NextResponse.json({ error: "no file provided" }, { status: 400 });
    }

    const upstream = new FormData();
    upstream.append("file", file, (file as File).name ?? "audio.wav");

    const headers: Record<string, string> = {};
    if (API_TOKEN) headers["Authorization"] = `Bearer ${API_TOKEN}`;

    const res = await fetch(`${API_URL}/api/detect`, {
      method: "POST",
      body: upstream,
      headers,
    });

    const data = await res.json().catch(() => ({ error: "bad upstream response" }));
    return NextResponse.json(data, { status: res.status });
  } catch (err: any) {
    return NextResponse.json(
      { error: `proxy error: ${err?.message ?? "unknown"}` },
      { status: 502 }
    );
  }
}
