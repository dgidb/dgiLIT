"""Network access is confined to an explicit, known-task source request."""
import xml.etree.ElementTree as ET


class PMCSourceProvider:
    def fetch(self, task):
        # Lazy imports: merely browsing never initializes retrieval or model code.
        from dgilit.pmc import PMCClient, NcbiTransport
        from dgilit.pubmed import _parse_pubmed_article
        transport = NcbiTransport(timeout=10, retries=1)
        client = PMCClient(transport)
        records = client.fetch_articles([task["pmcid"]])
        article = records.get(task["pmcid"])
        sections = [s.model_dump() for s in article.sections] if article else []
        errors = list(client.last_errors.values())
        original_source = (task.get("section_metadata") or {}).get("source")
        if original_source == "pubmed" and task.get("pmid"):
            try:
                root = ET.fromstring(transport.request("efetch.fcgi", db="pubmed", id=task["pmid"], retmode="xml"))
                for node in root.findall(".//PubmedArticle"):
                    record = _parse_pubmed_article(node)
                    if record and record.pmid == task["pmid"]:
                        sections.extend(s.model_dump() for s in record.abstract_sections)
            except (RuntimeError, ET.ParseError) as exc:
                errors.append(str(exc))
        return sections, errors
