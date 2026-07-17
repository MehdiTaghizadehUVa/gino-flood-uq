import "../globals.css";

export const metadata = {
  title: "FloodUQ Demo — Coastal flood uncertainty workspace",
  description: "Gated product demo for calibrated coastal flood uncertainty products",
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
