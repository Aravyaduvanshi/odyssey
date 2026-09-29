import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "OdysseyV1",
  description: "Edge navigation dashboard"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

