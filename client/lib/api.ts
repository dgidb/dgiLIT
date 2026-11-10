export type Cohort = "pilot" | "beyond_pilot";
export type ClaimRating = "up" | "down" | null;
export const ERROR_REASONS = [
  { code: "incorrect_gene_grouping", label: "Incorrect gene grouping" },
  { code: "incorrect_drug_grouping", label: "Incorrect drug grouping" },
  { code: "no_direct_interaction", label: "No direct interaction" },
  { code: "no_indirect_interaction", label: "No indirect interaction" },
  { code: "no_effects_on_gene_regulation", label: "No effects on gene regulation" },
  { code: "not_enough_evidence", label: "Not enough evidence" },
  { code: "incorrect_directionality", label: "Incorrect directionality" },
  { code: "incorrect_supporting_evidence", label: "Incorrect supporting evidence" },
] as const;
export type ErrorReason = typeof ERROR_REASONS[number]["code"];
export type ClaimAssessment = { rating: ClaimRating; rated_at: string | null; error_reasons: ErrorReason[] };
export type SourceResult = {
  state: "verified" | "unavailable" | "hash_mismatch" | "retrieval_error";
  origin: string; context: string | null; message: string; read_at: string;
  section_hash?: string; context_hash?: string;
};
export type ContextRow = ClaimAssessment & {
  task_id: string; pmcid: string; pmid: string | null; title: string | null;
  publication_year: string | null; coverage: string; article_status: string;
  drug: string; drug_id: string; gene: string; gene_id: string;
  evidence: string | null; evidence_verified: boolean | null; interaction_type: string | null;
  directionality: string | null; section_label: string | null; section_type: string | null;
  cohort: Cohort; updated_at: string;
};
export type ContextList = { items: ContextRow[]; total: number; page: number; page_size: number; read_at: string };
export type Metadata = {
  config?: Record<string, unknown>; execution_fingerprint?: {
    model_revisions?: Record<string, string>; packages?: Record<string, string>; prompt_hash?: string;
  }; pilot_report?: Record<string, unknown>; discovery_complete?: boolean;
  last_run?: { status?: string; pid?: number }; query?: string;
};
export type Summary = {
  read_at: string; dataset: string; positive_contexts: number; supporting_articles: number;
  cohorts: { cohort: Cohort; positive_contexts: number; supporting_articles: number }[];
  metadata: Metadata;
};
export type Entity = {
  id: string; text: string; entity_type: string; start: number; end: number; score?: number | null;
  source: string; concept?: { concept_id?: string; concept_label?: string; match_type?: string | number;
    normalizer?: string; error_message?: string | null };
};
export type Detail = ClaimAssessment & {
  read_at: string; dataset: string; metadata: Metadata; links: { pmc: string; pubmed?: string };
  prompt: { version?: string; saved_hash?: string; status: string; template?: string | null; note: string };
  entities: Entity[]; related_contexts: { status: string; contexts: number; positive: number; negative: number }[];
  task: {
    id: string; pmcid: string; pmid?: string; title?: string; publication_year?: string; license?: string;
    coverage: string; article_status: string; cohort: Cohort; is_current: boolean; status: string;
    drug: string; drug_id: string; gene: string; gene_id: string; article_hash: string;
    section_id: string; section_index: number; section_hash?: string; context_hash: string;
    context_start: number; context_end: number; config_hash: string; attempts: number;
    elapsed_seconds: number | null; updated_at: string;
    section_metadata?: { section_type?: string; label?: string; section_id?: string; parent_label?: string; source?: string };
    result: { evidence?: string; evidence_in_context?: boolean; interaction_type?: string; directionality?: string;
      model_id?: string; prompt_version?: string; generation_stats?: Record<string, unknown>; [key: string]: unknown };
    [key: string]: unknown;
  };
};
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/${path}`, { cache: "no-store", ...init });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body && typeof body === "object" && "detail" in body ? body.detail : null;
    throw new Error(typeof detail === "string" ? detail : "The data service could not complete this request.");
  }
  return body as T;
}
export const number = (value?: number) => new Intl.NumberFormat().format(value ?? 0);
export const when = (value: string) => new Date(value).toLocaleString(undefined, {dateStyle: "medium", timeStyle: "short"});
export const cohortLabel = (value: string) => value === "pilot" ? "Pilot" : "Beyond pilot";
export const coverageLabel = (value: string) => value === "full_text" ? "Full text" : value === "abstract_only" ? "Abstract" : value.replaceAll("_", " ");
