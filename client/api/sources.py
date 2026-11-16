from collections import OrderedDict
from copy import deepcopy
import gzip
import json
import re
import threading
import time

from .provenance import digest
from .repository import utcnow
from .upstream import PMCSourceProvider


class SourceBusy(RuntimeError):
    pass


class SourceService:
    def __init__(self, settings, provider=None):
        self.settings = settings
        self.provider = provider or PMCSourceProvider()
        self._fetch_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._maximum_bytes = 32 * 1024**2

    @staticmethod
    def _identity(task):
        if not re.fullmatch(r"PMC[0-9]+", task["pmcid"]):
            raise ValueError("Stored article identifier is not a valid PMCID")
        if task.get("pmid") and not re.fullmatch(r"[0-9]+", str(task["pmid"])):
            raise ValueError("Stored PubMed identifier is invalid")

    @staticmethod
    def verify(task, sections, origin):
        expected_section = task.get("section_hash")
        for section in sections:
            text = section.get("text")
            if not isinstance(text, str) or not expected_section or digest(text) != expected_section:
                continue
            start, end = task["context_start"], task["context_end"]
            if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end <= len(text):
                continue
            context = text[start:end]
            if digest(context) == task["context_hash"]:
                return {"state": "verified", "origin": origin, "context": context,
                        "section_hash": expected_section, "context_hash": task["context_hash"],
                        "read_at": utcnow(), "message": "Section and context hashes match the saved model input."}
        return {"state": "hash_mismatch", "origin": origin, "context": None, "read_at": utcnow(),
                "message": "Available text does not match the saved section and context hashes. It is not the original model input."}

    def _remember(self, task_id, value):
        size = len(json.dumps(value).encode())
        if size > self._maximum_bytes:
            return
        with self._cache_lock:
            old = self._cache.pop(task_id, None)
            if old:
                self._cache_bytes -= old[2]
            while self._cache and (len(self._cache) >= 100 or self._cache_bytes + size > self._maximum_bytes):
                _, item = self._cache.popitem(last=False)
                self._cache_bytes -= item[2]
            self._cache[task_id] = (time.monotonic(), deepcopy(value), size)
            self._cache_bytes += size

    def cached(self, task):
        self._identity(task)
        with self._cache_lock:
            item = self._cache.get(task["id"])
            if item and time.monotonic() - item[0] < self.settings.source_ttl:
                self._cache.move_to_end(task["id"])
                return deepcopy(item[1])
            if item:
                del self._cache[task["id"]]
                self._cache_bytes -= item[2]
        path = self.settings.data_dir / "text_cache" / f"{task['pmcid']}.json.gz"
        try:
            with gzip.open(path, "rt") as handle:
                article = json.load(handle)
            return self.verify(task, article.get("sections", []), "cli_cache")
        except FileNotFoundError:
            return {"state": "unavailable", "origin": "cli_cache", "context": None,
                    "message": "Full context is not in the CLI cache. Saved evidence and provenance remain available.", "read_at": utcnow()}
        except (OSError, ValueError, TypeError, AttributeError):
            return {"state": "retrieval_error", "origin": "cli_cache", "context": None,
                    "message": "The cached article could not be read. You can explicitly retrieve source text.", "read_at": utcnow()}

    def fetch(self, task):
        self._identity(task)
        cached = self.cached(task)
        if cached["state"] == "verified":
            return cached
        if not self._fetch_lock.acquire(blocking=False):
            raise SourceBusy("Another source retrieval is in progress. Try again shortly.")
        try:
            try:
                sections, errors = self.provider.fetch(task)
                if sections:
                    value = self.verify(task, sections, "pmc_retrieval")
                else:
                    value = {"state": "retrieval_error" if errors else "unavailable", "origin": "pmc_retrieval",
                             "context": None, "read_at": utcnow(), "message":
                             "Source retrieval failed. You can retry." if errors else "No accessible source text was returned."}
            except Exception:
                # Do not expose upstream request details or credentials to browsers.
                value = {"state": "retrieval_error", "origin": "pmc_retrieval", "context": None,
                         "read_at": utcnow(), "message": "Source retrieval failed. You can retry."}
            if value["state"] == "verified":
                self._remember(task["id"], value)
            return value
        finally:
            self._fetch_lock.release()
