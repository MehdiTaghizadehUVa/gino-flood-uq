import "./globals.css";

export const metadata = {
  title: "Coastal Flood-UQ Console",
  description: "Gated research flood inference with calibrated uncertainty"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
