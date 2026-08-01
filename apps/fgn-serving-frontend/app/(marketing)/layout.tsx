import type { Metadata, Viewport } from "next";
import { Inter, Inter_Tight, JetBrains_Mono } from "next/font/google";
import type { ReactNode } from "react";
import "./marketing.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-marketing-body", display: "swap" });
const interTight = Inter_Tight({ subsets: ["latin"], variable: "--font-marketing-display", display: "swap" });
const jetbrainsMono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-marketing-mono", display: "swap" });

export const metadata: Metadata = {
  metadataBase: new URL("https://flooduq.app"),
  title: "FloodUQ | Coastal flood probability and uncertainty modeling",
  description: "Compare coastal flood scenarios, estimate calibrated flood probability and timing, and see how closely plausible forecasts agree.",
  alternates: { canonical: "/" },
  icons: {
    icon: [{ url: "/marketing/brand/flooduq-app-icon.png", type: "image/png", sizes: "512x512" }],
    apple: [{ url: "/marketing/brand/flooduq-app-icon.png", sizes: "512x512", type: "image/png" }]
  },
  openGraph: {
    title: "FloodUQ | Know where flooding may go, and how closely forecasts agree",
    description: "Domain-specific coastal flood scenario modeling with fast probability, timing, forecast-range, and scenario-familiarity products.",
    url: "https://flooduq.app",
    siteName: "FloodUQ",
    type: "website",
    images: [{ url: "/marketing/og.jpg", width: 1200, height: 630, alt: "FloodUQ coastal flood probability and forecast-range maps" }]
  },
  twitter: {
    card: "summary_large_image",
    title: "FloodUQ | Coastal flood probability and uncertainty modeling",
    description: "Fast coastal flood scenario comparison with calibrated probabilities and visible forecast agreement.",
    images: ["/marketing/og.jpg"]
  }
};

export const viewport: Viewport = {
  colorScheme: "dark",
  themeColor: "#000000"
};

export default function MarketingLayout({ children }: { children: ReactNode }) {
  return (
    <html
      lang="en"
      data-theme="marketing"
      className={`${inter.variable} ${interTight.variable} ${jetbrainsMono.variable}`}
    >
      <body data-flooduq-marketing>{children}</body>
    </html>
  );
}
