import type { Metadata } from "next";
import { Outfit } from "next/font/google";
import "./globals.css";

const outfit = Outfit({
  subsets: ["latin"],
  variable: "--font-outfit",
  weight: ["300", "400", "500", "600", "700", "800"],
});

export const metadata: Metadata = {
  title: "Vi-SAFE: Real-Time AI Violence Detection Dashboard",
  description: "Advanced spatial-temporal AI CCTV monitoring for campus security",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${outfit.variable} h-full antialiased`}>
      <body className="min-h-full bg-[#07090e] text-[#e2e8f0] font-sans antialiased selection:bg-red-500/30">
        {children}
      </body>
    </html>
  );
}
