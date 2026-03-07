import type { Metadata } from "next";

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
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
