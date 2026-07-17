import type { Metadata, Viewport } from "next";
import { Inter, Inter_Tight, JetBrains_Mono } from "next/font/google";
import type { ReactNode } from "react";
import "./marketing.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-marketing-body", display: "swap" });
const interTight = Inter_Tight({ subsets: ["latin"], variable: "--font-marketing-display", display: "swap" });
const jetbrainsMono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-marketing-mono", display: "swap" });

export const metadata: Metadata = {
  metadataBase: new URL("https://flooduq.app"),
  title: "FloodUQ | Managed probabilistic coastal flood modeling",
  description: "A managed coastal flood intelligence service for rapid scenario evaluation with calibrated probabilities, explainable uncertainty, and monitored model behavior.",
  alternates: { canonical: "/" },
  openGraph: {
    title: "FloodUQ | Know where flooding may go, and how certain the model is",
    description: "Domain-specific coastal flood modeling with rapid ensemble inference, calibrated probability products, uncertainty source separation, and model monitoring.",
    url: "https://flooduq.app",
    siteName: "FloodUQ",
    type: "website",
    images: [{ url: "/marketing/og.jpg", width: 1200, height: 630, alt: "FloodUQ calibrated coastal flood uncertainty" }]
  },
  twitter: {
    card: "summary_large_image",
    title: "FloodUQ | Managed probabilistic coastal flood modeling",
    description: "Rapid, calibrated, and monitored coastal flood uncertainty for planning and scenario evaluation.",
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
