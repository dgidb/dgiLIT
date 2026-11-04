import { Suspense } from "react";
import { Browser } from "@/components/browser";
export default function Home() { return <Suspense fallback={<main className="shell main">Opening evidence browser…</main>}><Browser/></Suspense>; }
