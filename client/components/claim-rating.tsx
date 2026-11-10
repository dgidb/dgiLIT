"use client";
import { useRef, useState } from "react";
import { RefreshCw, ThumbsDown, ThumbsUp } from "lucide-react";
import { api, ClaimAssessment, ClaimRating, ERROR_REASONS, ErrorReason, when } from "@/lib/api";

export function RatingBadge({ rating }: { rating: ClaimRating }) {
  if (!rating) return null;
  const up = rating === "up";
  const Icon = up ? ThumbsUp : ThumbsDown;
  return <span className={`rating-badge rating-${rating}`} role="img"
    aria-label={`Shared assessment: thumbs ${rating}`} title={`Thumbs ${rating}`}>
    <Icon size={12} aria-hidden="true"/>
  </span>;
}

export function ClaimRatingControls({ taskId, assessment, onSaved }: {
  taskId: string; assessment: ClaimAssessment; onSaved: (value: ClaimAssessment) => void;
}) {
  const [saving, setSaving] = useState(false);
  const pending = useRef(false);
  const [error, setError] = useState("");
  async function save(rating: ClaimRating, error_reasons: ErrorReason[]) {
    if (pending.current) return;
    pending.current = true;
    setSaving(true); setError("");
    try {
      const saved = await api<ClaimAssessment>(`contexts/${taskId}/rating`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rating, error_reasons }),
      });
      onSaved(saved);
    } catch (e) {
      setError(e instanceof Error ? e.message : "The rating could not be saved. Try again.");
    } finally {
      pending.current = false;
      setSaving(false);
    }
  }
  function rate(value: Exclude<ClaimRating, null>) {
    void save(assessment.rating === value ? null : value, []);
  }
  function toggleReason(reason: ErrorReason) {
    const reasons = assessment.error_reasons.includes(reason)
      ? assessment.error_reasons.filter(code => code !== reason)
      : [...assessment.error_reasons, reason];
    void save("down", reasons);
  }
  return <div className="claim-assessment" aria-label="Shared claim assessment">
    <div className="assessment-label"><strong>Claim assessment</strong><span>Shared locally · independent of the AI prediction</span></div>
    <div className="rating-buttons" role="group" aria-label="Rate this claim">
      <button className={`rating-button rating-up ${assessment.rating === "up" ? "selected" : ""}`}
        type="button" aria-label="Thumbs up" aria-pressed={assessment.rating === "up"}
        disabled={saving} onClick={() => rate("up")}><ThumbsUp size={16} aria-hidden="true"/>Thumbs up</button>
      <button className={`rating-button rating-down ${assessment.rating === "down" ? "selected" : ""}`}
        type="button" aria-label="Thumbs down" aria-pressed={assessment.rating === "down"}
        disabled={saving} onClick={() => rate("down")}><ThumbsDown size={16} aria-hidden="true"/>Thumbs down</button>
    </div>
    {assessment.rating === "down" && <div className="error-reasons" role="group" aria-label="Error reasons">
      <strong>Error reasons</strong><p>Select all that apply (optional).</p>
      <div className="error-reason-buttons">
        {ERROR_REASONS.map(({ code, label }) => <button key={code} type="button"
          className={`error-reason-button ${assessment.error_reasons.includes(code) ? "selected" : ""}`}
          aria-pressed={assessment.error_reasons.includes(code)} disabled={saving}
          onClick={() => toggleReason(code)}>{label}</button>)}
      </div>
    </div>}
    <p className="assessment-status" role="status" aria-live="polite">{saving
      ? <><RefreshCw size={12} className="spin" aria-hidden="true"/>Saving assessment…</>
      : assessment.rating ? <>Thumbs {assessment.rating} saved{assessment.rated_at ? ` · ${when(assessment.rated_at)}` : ""}. Select it again to clear.</>
      : "No assessment yet. Choose thumbs up or down."}</p>
    {error && <p className="inline-error" role="alert">{error}</p>}
  </div>;
}
