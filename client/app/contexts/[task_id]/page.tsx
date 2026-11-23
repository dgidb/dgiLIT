import { Suspense } from "react";
import { ContextDetail } from "@/components/context-detail";
export default async function Page({ params }: { params: Promise<{ task_id: string }> }) {
  const { task_id } = await params;
  return <Suspense fallback={<main className="shell main">Opening context…</main>}><ContextDetail taskId={task_id}/></Suspense>;
}
