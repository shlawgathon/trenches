import type { Metadata } from "next";
import Script from "next/script";

import "./globals.css";
import "../src/styles.css";
import "../src/components/monitoring.css";
import "../src/components/command-map.css";
import "mapbox-gl/dist/mapbox-gl.css";

export const metadata: Metadata = {
  title: "Trenches Monitoring Deck",
  description: "Fog-of-war monitoring dashboard with Mapbox-powered theater view.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const publicEnv = {
    apiBaseUrl: process.env.NEXT_PUBLIC_API_BASE_URL || process.env.API_BASE_URL || "http://localhost:8000",
    vercelApiBase: process.env.NEXT_PUBLIC_VERCEL_API_BASE || "/api",
    enableSourceLogic: process.env.NEXT_PUBLIC_ENABLE_SOURCE_LOGIC || "false",
    mapboxToken: process.env.NEXT_PUBLIC_MAPBOX_TOKEN || process.env.MAPBOX_TOKEN || "",
  };

  return (
    <html lang="en">
      <body>
        <Script
          id="trenches-runtime-env"
          strategy="beforeInteractive"
          dangerouslySetInnerHTML={{
            __html: `window.__trenchesEnv = ${JSON.stringify(publicEnv)};`,
          }}
        />
        {children}
      </body>
    </html>
  );
}
