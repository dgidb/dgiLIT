"use client";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { ArrowUpRight, ChevronLeft, ChevronRight, Database, Dna, RefreshCw, Search, ShieldCheck } from "lucide-react";
import { api, cohortLabel, ContextList, coverageLabel, number, Summary, when } from "@/lib/api";

export function Brand() {
  return <Link href="/" className="brand" aria-label="dgiLIT home"><span className="brand-mark"><Dna size={23}/></span><span>dgi<span className="brand-light">LIT</span><small>Evidence explorer</small></span></Link>;
}
export function Header() {
  return <header className="site-header"><div className="shell header-inner"><Brand/><div className="header-right"><span className="active-nav">Explore contexts</span><span className="local-tag"><span/>Local workspace</span></div></div></header>;
}
export function Browser() {
  const router = useRouter();
  const params = useSearchParams();
  const query = params.toString();
  const [nonce, setNonce] = useState(0);
  const [data, setData] = useState<ContextList | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setError("");
    api<ContextList>(`contexts?${query}`, { signal: controller.signal }).then(setData)
      .catch(e => { if (e.name !== "AbortError") setError(e.message); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    api<Summary>(`summary${nonce ? "?refresh=true" : ""}`, { signal: controller.signal }).then(setSummary).catch(() => {});
    return () => controller.abort();
  }, [query, nonce]);
  const pilot = summary?.cohorts.find(c => c.cohort === "pilot")?.positive_contexts ?? 0;
  const beyond = summary?.cohorts.find(c => c.cohort === "beyond_pilot")?.positive_contexts ?? 0;
  const page = data?.page ?? Number(params.get("page") || 1);
  const pageSize = data?.page_size ?? 50;
  const start = data?.total ? (page - 1) * pageSize + 1 : 0;
  const end = data ? Math.min(page * pageSize, data.total) : 0;
  function changePage(next: number) { const q = new URLSearchParams(query); q.set("page", String(next)); router.push(`/?${q}`); }
  function apply(form: FormData) {
    const q = new URLSearchParams();
    for (const key of ["drug", "gene", "article", "cohort", "coverage"]) {
      const value = String(form.get(key) || "").trim();
      if (value && value !== "all") q.set(key, value);
    }
    router.push(q.size ? `/?${q}` : "/");
  }
  return <><Header/><main className="shell main">
    <div className="page-heading"><div><div className="eyebrow">DRUG–GENE EVIDENCE</div><h1>Explore discovered interactions</h1><p className="lead">Follow each AI-supported interaction back to its evidence, paper, and model.</p></div>
      <button className="button secondary" onClick={() => setNonce(n => n + 1)} disabled={loading}><RefreshCw size={15} className={loading ? "spin" : ""}/>Refresh data</button></div>
    <section className="metrics" aria-label="All processed data">
      <Metric label="Positive contexts" value={summary ? number(summary.positive_contexts) : "—"} note="Across all processed articles" icon={<ShieldCheck size={17}/>}/>
      <Metric label="Supporting articles" value={summary ? number(summary.supporting_articles) : "—"} note="Distinct papers with positive evidence" icon={<Database size={17}/>}/>
      <Metric label="Pilot contexts" value={summary ? number(pilot) : "—"} note="Initial stratified sample"/>
      <Metric label="Beyond the pilot" value={summary ? number(beyond) : "—"} note="Evidence from the larger cohort"/>
    </section>
    <section className="results-panel">
      <div className="panel-heading"><div><h2>Positive evidence</h2><p>Both cohorts included by default. Each row is one pair in one text context.</p></div><span className="subtle-tag">Read-only checkpoint access</span></div>
      <form className="filters" key={query} onSubmit={e => { e.preventDefault(); apply(new FormData(e.currentTarget)); }}>
        <label>Drug<input name="drug" defaultValue={params.get("drug") || ""} placeholder="Name or concept ID"/></label>
        <label>Gene<input name="gene" defaultValue={params.get("gene") || ""} placeholder="e.g. EGFR"/></label>
        <label className="article-filter">Article<input name="article" defaultValue={params.get("article") || ""} placeholder="Title, PMCID, or PMID"/></label>
        <label>Cohort<select aria-label="Cohort" name="cohort" defaultValue={params.get("cohort") || "all"}><option value="all">All articles</option><option value="pilot">Pilot only</option><option value="beyond_pilot">Beyond pilot</option></select></label>
        <label>Coverage<select aria-label="Coverage" name="coverage" defaultValue={params.get("coverage") || "all"}><option value="all">All sources</option><option value="full_text">Full text</option><option value="abstract_only">Abstract only</option></select></label>
        <button type="submit" className="button primary"><Search size={15}/>Search</button>
        {query && <Link href="/" className="reset">Reset</Link>}
      </form>
      <div className="table-status" aria-live="polite"><span>{loading ? "Reading saved results…" : data ? `${number(data.total)} matching contexts` : "Results unavailable"}</span><span>{data && `Read ${when(data.read_at)}`}</span></div>
      {error ? <div role="alert" className="empty-state error-state"><h3>We couldn’t read the results</h3><p>{error}</p><button className="button secondary" onClick={() => setNonce(n => n + 1)}>Try again</button></div> : !data && loading ? <div className="empty-state"><RefreshCw className="spin"/><p>Opening the saved evidence…</p></div> : data?.items.length === 0 ? <div className="empty-state"><Search size={27}/><h3>No contexts match this view</h3><p>Try another filter, return to the first page, or refresh as the extraction progresses.</p><Link href="/" className="button secondary">View all contexts</Link></div> : <div className={`table-scroll ${loading ? "refreshing" : ""}`} aria-busy={loading}><table className="evidence-table"><thead><tr><th>Drug / gene</th><th>Supporting evidence</th><th>Source article</th><th>Cohort</th><th>Saved</th><th><span className="sr-only">Entry details</span></th></tr></thead><tbody>{data?.items.map(row => <tr key={row.task_id}>
        <td><strong className="entity-name">{row.drug}</strong><span className="gene-pill">{row.gene}</span><small className="cell-note">{row.interaction_type || "Interaction supported"}</small></td>
        <td className="evidence-cell"><p className="clamp-3">{row.evidence || "Evidence quote unavailable"}</p><span className="section-note">{row.section_label || row.section_type || "Section unspecified"}</span></td>
        <td className="article-cell"><span className="mono article-id">{row.pmcid}</span><span className="clamp-2 article-title">{row.title || "Untitled article"}</span><small>{coverageLabel(row.coverage)}{row.publication_year ? ` · ${row.publication_year}` : ""}</small></td>
        <td><span className={`cohort-tag ${row.cohort === "pilot" ? "pilot" : "beyond"}`}>{cohortLabel(row.cohort)}</span></td>
        <td className="date-cell" title={row.updated_at}>{when(row.updated_at)}</td>
        <td><Link className="detail-link" aria-label={`View evidence for ${row.drug} and ${row.gene} in ${row.pmcid}`} href={`/contexts/${row.task_id}?returnTo=${encodeURIComponent(query ? `/?${query}` : "/")}`}>View<ArrowUpRight size={14}/></Link></td>
      </tr>)}</tbody></table></div>}
      <div className="pagination"><span>{data ? `${number(start)}–${number(end)} of ${number(data.total)} contexts` : ""}</span><div><button className="page-button" disabled={loading || page <= 1} onClick={() => changePage(page - 1)} aria-label="Previous page"><ChevronLeft size={17}/></button><span>Page {number(page)}</span><button className="page-button" disabled={loading || !data || page * pageSize >= data.total} onClick={() => changePage(page + 1)} aria-label="Next page"><ChevronRight size={17}/></button></div></div>
    </section>
    <footer className="page-footer"><p><span className="live-dot"/>Saved results from an ongoing extraction. Refresh to load new evidence.</p><p>AI predictions for expert review · {summary ? `Summary read ${when(summary.read_at)}` : "Summary loading"}</p></footer>
  </main></>;
}
function Metric({ label, value, note, icon }: { label: string; value: string; note: string; icon?: React.ReactNode }) {
  return <article className="metric"><div className="metric-label">{label}{icon}</div><strong>{value}</strong><p>{note}</p></article>;
}
