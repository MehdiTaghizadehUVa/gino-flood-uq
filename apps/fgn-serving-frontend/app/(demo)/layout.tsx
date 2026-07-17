import "../globals.css";

export const metadata = {
  title: "FloodUQ Demo — Coastal flood uncertainty workspace",
  description: "Gated product demo for calibrated coastal flood uncertainty products",
  icons: {
    icon: [{ url: "/marketing/brand/flooduq-app-icon.png", type: "image/png", sizes: "512x512" }],
    apple: [{ url: "/marketing/brand/flooduq-app-icon.png", sizes: "512x512", type: "image/png" }]
  },
  robots: {
    index: false,
    follow: false
  }
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body data-flooduq-demo>{children}</body>
    </html>
  );
}
