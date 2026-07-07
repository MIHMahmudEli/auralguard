# AuralGuard UI (Next.js)

End-user web app: record or upload a voice clip and get an AI-generated / human verdict.
Calls the Hugging Face Space API through a server-side proxy (`app/api/detect/route.ts`)
so the API URL/token stay private.

## Setup
```bash
cd ui
npm install
cp .env.example .env.local     # set AURALGUARD_API_URL to your HF Space
npm run dev                    # http://localhost:3000
```

## Deploy (Vercel)
1. Import the `ui/` directory as a Vercel project.
2. Set `AURALGUARD_API_URL` (and `AURALGUARD_API_TOKEN` if the Space is private) in the
   Vercel environment variables.
3. Deploy.

## Structure
```
app/
  layout.tsx           # root layout + metadata
  page.tsx             # recorder + uploader + result (client)
  globals.css          # Tailwind
  api/detect/route.ts  # server proxy → HF Space /api/detect
components/
  ResultCard.tsx       # verdict UI
```

The UI is intentionally framework-light (no state library, no chart deps) so it stays
easy to audit and fast to deploy. To add the spectro-temporal attribution heatmap, render
the `heatmap_png_base64` field from the API response inside `ResultCard`.
