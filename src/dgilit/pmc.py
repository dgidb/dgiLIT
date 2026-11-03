"""PMC-native discovery and retrieval, including records without a PMID."""
from __future__ import annotations

from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from pydantic import BaseModel, ConfigDict, Field

from .pubmed import PubMedArticleSection, _parse_pmc_sections, _parse_pubmed_article

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


class RequestLimiter:
    """Shared across all NCBI calls made by one pipeline process."""

    def __init__(self, interval: float = 0.35):
        self.interval = max(interval, 1 / 3)
        self.last = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            time.sleep(max(0.0, self.last + self.interval - time.monotonic()))
            self.last = time.monotonic()


_LIMITER = RequestLimiter()


class NcbiTransport:
    def __init__(self, email=None, timeout=60, retries=5, limiter=None):
        self.email = email or os.getenv("MC_EMAIL") or os.getenv("NCBI_EMAIL")
        self.timeout, self.retries = timeout, retries
        self.limiter = limiter or _LIMITER

    def request(self, endpoint, **params):
        params.update(tool="dgilit")
        if self.email:
            params["email"] = self.email
        if os.getenv("NCBI_API_KEY"):
            params["api_key"] = os.environ["NCBI_API_KEY"]
        request = urllib.request.Request(
            EUTILS + endpoint, data=urllib.parse.urlencode(params).encode(),
            headers={"User-Agent": "dgilit/0.1.0"},
        )
        for attempt in range(self.retries + 1):
            self.limiter.wait()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return response.read()
            except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead) as exc:
                transient = not isinstance(exc, urllib.error.HTTPError) or exc.code in {408, 429, 500, 502, 503, 504}
                if not transient or attempt == self.retries:
                    # Do not persist request URLs containing contact/API-key parameters.
                    raise RuntimeError(f"NCBI {endpoint}: {type(exc).__name__}, HTTP {getattr(exc, 'code', 'n/a')}") from exc
                delay = min(60, 2 ** attempt)
                if isinstance(exc, urllib.error.HTTPError) and exc.headers:
                    retry_after = exc.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            try:
                                delay = max(delay, (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds())
                            except (TypeError, ValueError):
                                pass
                time.sleep(max(0, delay))


class PMCArticle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pmcid: str
    pmid: str | None = None
    title: str | None = None
    journal: str | None = None
    publication_year: str | None = None
    license: str | None = None
    full_text_available: bool = False
    sections: list[PubMedArticleSection] = Field(default_factory=list)

    @property
    def coverage(self):
        return "full_text" if self.full_text_available else "abstract_only" if self.sections else "no_text"


def parse_pmc_article(article: ET.Element) -> PMCArticle:
    # JATS documents may have a default namespace.
    for element in article.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    meta = article.find("./front/article-meta")
    if meta is None:
        raise ValueError("PMC XML has no article metadata")
    ids = {x.get("pub-id-type"): (x.text or "").strip() for x in meta.findall("article-id")}
    uid = next((ids[k] for k in ["pmc", "pmcid", "pmcaid", "pmcaiid"] if ids.get(k)), None)
    if not uid:
        raise ValueError("PMC XML has no PMCID")
    def text(path):
        node = article.find(path)
        return " ".join(node.itertext()).strip() if node is not None else None
    sections = _parse_pmc_sections(article)
    # Some JATS bodies contain direct paragraphs, not <sec> elements.
    body = article.find("./body")
    if body is not None:
        direct = [" ".join(child.itertext()).strip() for child in body if child.tag != "sec"]
        direct = [x for x in direct if x]
        if direct:
            sections.append(PubMedArticleSection(text="\n".join(direct), section_type="body", source="pmc"))
    years = meta.findall("pub-date/year")
    return PMCArticle(
        pmcid="PMC" + uid.removeprefix("PMC"), pmid=ids.get("pmid") or None,
        title=text("./front/article-meta/title-group/article-title"),
        journal=text("./front/journal-meta/journal-title-group/journal-title"),
        publication_year=years[0].text if years else None,
        license=text("./front/article-meta/permissions/license"), sections=sections,
        full_text_available=any(s.section_type != "abstract" for s in sections),
    )


class PMCClient:
    """Uses supported E-utilities endpoints; does not scrape article webpages."""

    def __init__(self, transport=None):
        self.transport = transport or NcbiTransport()
        self.last_errors = {}

    @staticmethod
    def query(start: date, end: date):
        return f'"{start:%Y/%m/%d}"[dp] : "{end:%Y/%m/%d}"[dp]'

    def search(self, start, end, offset=0, size=0, query=None):
        payload = json.loads(self.transport.request(
            "esearch.fcgi", db="pmc", term=query or self.query(start, end),
            retmode="json", retstart=offset, retmax=size, sort="uid",
        ))
        result = payload.get("esearchresult", {})
        if "count" not in result or result.get("ERROR") or result.get("errorlist"):
            raise ValueError(f"PMC search returned an error: {result}")
        return int(result["count"]), ["PMC" + str(x).removeprefix("PMC") for x in result.get("idlist", [])]

    def summaries(self, pmcids):
        payload = json.loads(self.transport.request(
            "esummary.fcgi", db="pmc", id=",".join(x.removeprefix("PMC") for x in pmcids), retmode="json",
        ))["result"]
        return {"PMC" + uid: payload[uid] for uid in payload.get("uids", [])}

    def _fetch(self, pmcids):
        root = ET.fromstring(self.transport.request(
            "efetch.fcgi", db="pmc", id=",".join(x.removeprefix("PMC") for x in pmcids), retmode="xml",
        ))
        for element in root.iter():
            element.tag = element.tag.rsplit("}", 1)[-1]
        nodes = [root] if root.tag == "article" else root.findall(".//article")
        return {a.pmcid: a for a in (parse_pmc_article(node) for node in nodes)}

    def fetch_articles(self, pmcids):
        """Return retrieved records; per-record failures are in ``last_errors``.

        Missing/full-text-denied records still get metadata and abstract fallback.
        A failed batch is isolated into individual requests so it cannot lose peers.
        """
        pmcids = list(dict.fromkeys(pmcids))
        self.last_errors = {}
        articles = {}
        for start in range(0, len(pmcids), 25):
            batch = pmcids[start:start + 25]
            try:
                articles.update(self._fetch(batch))
            except (RuntimeError, ValueError, ET.ParseError):
                pass
            for pmcid in batch:
                if pmcid in articles:
                    continue
                error = None
                try:
                    articles.update(self._fetch([pmcid]))
                except (RuntimeError, ValueError, ET.ParseError) as exc:
                    error = str(exc)
                if pmcid not in articles:
                    try:
                        summary = self.summaries([pmcid]).get(pmcid)
                        if summary is None:
                            raise ValueError("PMC metadata was not returned")
                        ids = summary.get("articleids", [])
                        pmid = next((str(x["value"]) for x in ids if x.get("idtype") == "pmid"), None)
                        article = PMCArticle(pmcid=pmcid, pmid=pmid, title=summary.get("title"),
                                             publication_year=str(summary.get("pubdate", ""))[:4] or None)
                        if pmid:
                            root = ET.fromstring(self.transport.request("efetch.fcgi", db="pubmed", id=pmid, retmode="xml"))
                            for node in root.findall(".//PubmedArticle"):
                                pubmed = _parse_pubmed_article(node)
                                if pubmed and pubmed.pmid == pmid:
                                    article.sections = pubmed.abstract_sections
                        articles[pmcid] = article
                    except (RuntimeError, ValueError, ET.ParseError) as exc:
                        self.last_errors[pmcid] = str(exc)
                # An unsuccessful request is not evidence of definitive unavailability.
                if error:
                    self.last_errors[pmcid] = error
        return articles
