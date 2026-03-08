import type { Metadata } from "next";

import "./globals.css";
import "mapbox-gl/dist/mapbox-gl.css";

export const metadata: Metadata = {
  title: "Trenches — Fog of War Diplomacy Simulator",
  description:
    "Multi-agent geopolitical crisis simulator with live intelligence feeds and Mapbox globe visualization.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen overflow-hidden antialiased">
        {children}
      </body>
    </html>
  );
}
