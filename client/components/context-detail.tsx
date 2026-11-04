"use client";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { ArrowLeft, Check, Copy, Download, ExternalLink, FileText, RefreshCw, ShieldCheck } from "lucide-react";
import { api, cohortLabel, coverageLabel, Detail, SourceResult, when } from "@/lib/api";
import { Header } from "./browser";

export function ContextDetail({ taskId }: { taskId: string }) {
  const params = useSearchParams();
  const rawReturn = params.get("returnTo") || "/";
  const returnTo = rawReturn === "/" || rawReturn.startsWith("/?") ? rawReturn : "/";
  const [detail, setDetail] = useState<Detail | null>(null);
  const [source, setSource] = useState<SourceResult | null>(null);
  const [error, setError] = useState("");
  const [sourceError, setSourceError] = useState("");
  const [fetching, setFetching] = useState(false);
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    setError(""); setDetail(null); setSource(null);
    api<Detail>(`contexts/${taskId}`, { signal: controller.signal }).then(value => {
      setDetail(value);
      return api<SourceResult>(`contexts/${taskId}/source`, { signal: controller.signal }).then(setSource)
        .catch(e => { if (e.name !== "AbortError") setSourceError(e.message); });
    }).catch(e => { if (e.name !== "AbortError") setError(e.message); });
    return () => controller.abort();
  }, [taskId]);
  async function fetchSource() {
    setFetching(true); setSourceError("");
    try { setSource(await api<SourceResult>(`contexts/${taskId}/source`, { method: "POST" })); }
    catch (e) { setSourceError(e instanceof Error ? e.message : "Source retrieval failed."); }
    finally { setFetching(false); }
  }
  async function copy() {
    try { await navigator.clipboard.writeText(taskId); setCopied(true); }
    catch { setSourceError("Clipboard access was unavailable. The full task ID is shown in Provenance below."); }
  }
  function download() {
    const blob = new Blob([JSON.stringify({ ...detail, source_context: source }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = `${detail?.task.pmcid}-${taskId.slice(0,12)}-provenance.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const t = detail?.task;
  const r = t?.result;
  const fingerprint = detail?.metadata.execution_fingerprint;
  return <><Header/><main className="shell main detail-main"><Link href={returnTo} className="back-link"><ArrowLeft size={15}/>Back to results</Link>
    {error ? <div className="empty-state error-state" role="alert"><h1>Context unavailable</h1><p>{error}</p><Link href={returnTo} className="button secondary">Return to results</Link></div> : !detail || !t || !r ? <div className="empty-state"><RefreshCw className="spin"/><p>Reading saved provenance…</p></div> : <>
      <div className="detail-heading"><div><div className="eyebrow">POSITIVE CONTEXT</div><h1>{t.drug}<span className="pair-divider">/</span>{t.gene}</h1><div className="detail-tags"><span className={`cohort-tag ${t.cohort === "pilot" ? "pilot" : "beyond"}`}>{cohortLabel(t.cohort)}</span><span className="subtle-tag">{coverageLabel(t.coverage)}</span><span className="subtle-tag">{String(r.interaction_type || "Interaction supported")}</span><span className="muted">Saved {when(t.updated_at)}</span></div></div>
        <div className="actions"><button className="button secondary" onClick={copy}>{copied ? <Check size={15}/> : <Copy size={15}/>} {copied ? "Copied" : "Copy task ID"}</button><button className="button primary" onClick={download}><Download size={15}/>Provenance JSON</button></div></div>
      {!t.is_current && <div className="notice">Historical entry: the CLI now has different text for this article. This page preserves the original saved task; it is excluded from current browse results.</div>}
      <div className="detail-layout"><div className="detail-content">
        <section className="card evidence-card"><div className="section-heading"><h2>Supporting evidence</h2><span className={`verification ${r.evidence_in_context ? "verified" : ""}`}><ShieldCheck size={14}/>{r.evidence_in_context ? "Quote verified when classified" : "Verification not recorded"}</span></div>
          <blockquote>{String(r.evidence || "No supporting quote was saved.")}</blockquote><div className="quote-caption"><FileText size={14}/>{t.section_metadata?.label || t.section_metadata?.section_type || "Section unspecified"}{t.section_metadata?.parent_label ? ` · ${t.section_metadata.parent_label}` : ""}</div>
          <MetaGrid rows={[["Requested drug", `${t.drug} · ${t.drug_id}`], ["Requested gene", `${t.gene} · ${t.gene_id}`], ["Interaction type", r.interaction_type], ["Directionality", r.directionality]]}/>
        </section>
        <section className="card"><div className="section-heading"><h2>Source context</h2>{source?.state === "verified" && <span className="verification verified"><ShieldCheck size={14}/>Hashes verified</span>}</div>
          <p className="section-description">The exact text assessed for this pair, when a matching source is available.</p>
          {source?.state === "verified" ? <><div className="source-status">{source.origin === "cli_cache" ? "Read from the CLI’s saved text cache" : "Retrieved from PMC and matched to the saved hashes"} · {when(source.read_at)}</div><div className="source-text">{source.context}</div></> : <div className="source-empty"><FileText size={25}/><p>{source?.message || (sourceError ? "Cached source could not be read." : "Checking the existing text cache…")}</p><p className="muted">Retrieval is explicit. Changed text will never be shown as the original model input.</p><button className="button secondary" onClick={fetchSource} disabled={fetching || (!source && !sourceError)}><RefreshCw size={15} className={fetching ? "spin" : ""}/>{fetching ? "Retrieving from PMC…" : "Fetch source context"}</button></div>}
          {sourceError && <p className="inline-error" role="alert">{sourceError}</p>}
        </section>
        <section className="card"><h2>Detected entity mentions</h2><p className="section-description">NER mentions grounded to the two requested concepts in this section.</p>{detail.entities.length ? <div className="table-scroll"><table className="entity-table"><thead><tr><th>Mention / type</th><th>Normalized concept</th><th>Span / NER score</th></tr></thead><tbody>{detail.entities.map(e => <tr key={e.id}><td><strong>{e.text}</strong><small>{e.entity_type}</small></td><td>{e.concept?.concept_label}<small className="mono">{e.concept?.concept_id}</small></td><td className="mono">{e.start}–{e.end}<small>{e.score == null ? "Score not recorded" : e.score.toFixed(3)}</small></td></tr>)}</tbody></table></div> : <p className="muted">Matching mention records were not saved.</p>}<details><summary>Complete mention and normalization records</summary><Json value={detail.entities}/></details></section>
        <section className="card"><h2>Model and prompt</h2><MetaGrid rows={[["Model", r.model_id], ["Model revision", fingerprint?.model_revisions?.[String(r.model_id)]], ["Prompt version", detail.prompt.version], ["Template verification", detail.prompt.status.replaceAll("_", " ")], ["Saved prompt hash", detail.prompt.saved_hash], ["Thinking enabled", r.enable_thinking], ["Temperature", r.temperature], ["Elapsed seconds", t.elapsed_seconds]]}/><p className="section-description">{detail.prompt.note}</p>
          <details><summary>Generation statistics</summary><Json value={r.generation_stats || {}}/></details>
          <details><summary>Model revisions and package versions</summary><Json value={fingerprint || {}}/></details>
          {detail.prompt.template && <details><summary>Verified prompt template</summary><pre className="code-block">{detail.prompt.template}</pre></details>}
          <details><summary>Complete saved classifier result</summary><Json value={r}/></details>
        </section>
        <section className="card"><h2>Provenance</h2><MetaGrid rows={[["Task ID", t.id], ["Task state", t.status], ["Attempts", t.attempts], ["Configuration fingerprint", t.config_hash], ["Article text hash", t.article_hash], ["Section hash", t.section_hash], ["Context hash", t.context_hash], ["Dataset", detail.dataset], ["Detail read at", detail.read_at]]}/><details><summary>Run configuration and discovery query</summary><Json value={{config: detail.metadata.config, query: detail.metadata.query}}/></details></section>
      </div><aside className="detail-sidebar"><section className="card article-card"><div className="eyebrow">SOURCE PAPER</div><h2>{t.title || "Untitled article"}</h2><p className="mono article-id">{t.pmcid}{t.pmid ? ` · PMID ${t.pmid}` : ""}</p><div className="source-links"><a href={detail.links.pmc} target="_blank" rel="noreferrer">Open in PMC<ExternalLink size={13}/></a>{detail.links.pubmed && <a href={detail.links.pubmed} target="_blank" rel="noreferrer">PubMed<ExternalLink size={13}/></a>}</div><MetaGrid rows={[["Publication year", t.publication_year], ["Coverage", coverageLabel(t.coverage)], ["Article processing", t.article_status.replaceAll("_", " ")], ["License", t.license]]}/></section>
        <section className="card"><h2>Section provenance</h2><MetaGrid rows={[["Type", t.section_metadata?.section_type], ["Label", t.section_metadata?.label], ["Parent", t.section_metadata?.parent_label], ["Source", t.section_metadata?.source], ["Source section ID", t.section_metadata?.section_id], ["Section index", t.section_index], ["Context character offsets", `${t.context_start}–${t.context_end}`], ["Internal section ID", t.section_id]]}/></section>
        <section className="card"><h2>This article pair</h2><p className="section-description">All saved contexts for this drug/gene pair in the same article version.</p><table className="compact-table"><thead><tr><th>State</th><th>Contexts</th><th>Positive</th><th>Negative</th></tr></thead><tbody>{detail.related_contexts.map(row => <tr key={row.status}><td>{row.status}</td><td>{row.contexts}</td><td>{row.positive}</td><td>{row.negative}</td></tr>)}</tbody></table><p className="muted small">A positive context can be available while other work remains unfinished.</p></section>
      </aside></div><footer className="page-footer"><p>AI-discovered evidence for expert review.</p><Link href={returnTo}>Back to filtered results</Link></footer>
    </>}
  </main></>;
}
function MetaGrid({rows}: {rows: [string, unknown][]}) {
  return <dl className="meta-grid">{rows.map(([key,value]) => <div key={key}><dt>{key}</dt><dd className={key.toLowerCase().includes("hash") || key.includes("ID") || key.includes("fingerprint") ? "mono" : ""}>{value == null || value === "" ? <span className="muted">Not recorded</span> : String(value)}</dd></div>)}</dl>;
}
function Json({value}: {value: unknown}) { return <pre className="code-block">{JSON.stringify(value,null,2)}</pre>; }
