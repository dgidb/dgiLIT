import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
  title: "dgiLIT · Claim Explorer",
  description: "Browse AI-discovered drug–gene contexts with their complete saved provenance.",
  icons: { icon: "/favicon.svg" },
};
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
