import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AuralGuard — AI Voice Detector",
  description: "Detect whether a voice recording is AI-generated.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
