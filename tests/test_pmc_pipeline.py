from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock
import urllib.error
import xml.etree.ElementTree as ET

from dgilit.pmc import PMCArticle, PMCClient, NcbiTransport, parse_pmc_article
from dgilit.pmc_pipeline import PMCPipeline, PipelineConfig
from dgilit.pmc_store import Store, ResourceLimit, digest
from dgilit.pubmed import PubMedArticleSection
from dgilit.tagging import EntityMention, NormalizedConcept, TaggedTextBlock
from dgilit.tagging.normalizers import SQLiteNormalizerCache, ViccNormalizer
from dgilit.tagging.taggers import BioBertEntityTagger, TaggerConfig
from dgilit.text_windows import text_windows


class CharacterTokenizer:
    model_max_length = 8

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i+1) for i in range(len(text))]}

    def num_special_tokens_to_add(self, **kwargs):
        return 2


class FakeClient:
    query = staticmethod(PMCClient.query)

    def __init__(self, articles):
        self.articles = articles
        self.last_errors = {}
        self.fetches = 0

    def search(self, start, end, offset=0, size=0):
        ids = sorted(self.articles)
        return len(ids), ids[offset:offset+size]

    def fetch_articles(self, ids):
        self.fetches += 1
        return {i: self.articles[i] for i in ids}


class FakeTagger:
    def tag_block(self, text, pmid, block_id):
        mentions = [EntityMention(text=label, entity_type=kind, start=0, end=4,
                    concept=NormalizedConcept(concept_id=identifier, concept_label=label))
                    for kind, identifier, label in [("drug", "drug:1", "Drug"), ("gene", "gene:1", "GENE")]]
        return TaggedTextBlock(context=text, pmid=pmid, block_id=block_id, entities=mentions)


class FakeLLM:
    last_generation_stats = {"prompt_tokens": 25, "generation_tokens": 10}

    def __init__(self, outcomes=None):
        self.outcomes = iter(outcomes or [])
        self.calls = 0

    def context_windows(self, prompt, text, drug, gene, schema, overlap):
        return [(0, len(text))]

    def invoke_json(self, system, user, schema):
        self.calls += 1
        outcome = next(self.outcomes, True)
        if isinstance(outcome, Exception):
            raise outcome
        evidence = user.split("Context:\n", 1)[1].strip()
        return SimpleNamespace(parsed_json={"drug": "Drug", "gene": "GENE", "interaction": outcome,
                                            "evidence": evidence, "interaction_type": "inhibition", "directionality": "inhibiting"})


def article(pmcid="PMC1", text="Drug inhibits GENE."):
    return PMCArticle(pmcid=pmcid, publication_year="2020", sections=[
        PubMedArticleSection(text=text, section_type="results", source="pmc")], full_text_available=True)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.revision = patch("dgilit.pmc_pipeline._model_revision", return_value="test-revision")
        self.revision.start()
        self.addCleanup(self.revision.stop)

    def pipeline(self, articles=None, llm=None):
        return PMCPipeline(self.root, PipelineConfig(start="2020-01-01", end="2020-12-31"),
                           client=FakeClient(articles or {"PMC1": article()}), pretagger=FakeTagger(),
                           llm=llm or FakeLLM(), min_free_bytes=0, cache_bytes=100000)

    def test_complete_resume_and_parquet(self):
        llm = FakeLLM()
        p = self.pipeline(llm=llm)
        status = p.run()
        self.assertTrue(status["successful_completion"])
        self.assertEqual(llm.calls, 1)
        p.run()
        self.assertEqual(llm.calls, 1)
        exported = p.export()
        self.assertEqual(exported["datasets"]["positive_interactions"]["rows"], 1)
        import polars as pl
        generation = exported["generation"]
        positive = pl.read_parquet(self.root / "exports" / generation / "positive_interactions" / "*.parquet")
        self.assertIsNone(positive["pmid"][0])
        classifications = pl.read_parquet(self.root / "exports" / generation / "classifications" / "*.parquet")
        self.assertTrue(classifications["interaction"][0])
        self.assertEqual(classifications["evidence"][0], "Drug inhibits GENE.")

    def test_failure_is_not_negative_retry_reuses_success(self):
        llm = FakeLLM([True, RuntimeError("transient"), False])
        p = self.pipeline({"PMC1": article(), "PMC2": article("PMC2")}, llm)
        status = p.run()
        self.assertEqual(status["tasks"], {"done": 1, "failed": 1})
        self.assertFalse(status["successful_completion"])
        p.retry_failed()
        p.run()
        self.assertEqual(llm.calls, 3)
        self.assertTrue(p.status()["successful_completion"])
        p.export()

    def test_crash_recovery_and_configuration_guard(self):
        p = self.pipeline()
        p.run()
        with p.store.db:
            p.store.db.execute("UPDATE tasks SET status='running'")
        with p.store.writer_lock():
            self.assertEqual(p.store.db.execute("SELECT status FROM tasks").fetchone()[0], "pending")
        with self.assertRaises(ValueError):
            PMCPipeline(self.root, PipelineConfig(start="2021-01-01", end="2021-12-31"))

    def test_count_mismatch_leaves_discovery_incomplete(self):
        p = self.pipeline()
        p.client.search = lambda start, end, offset=0, size=0: (2, ["PMC1"] if size else [])
        with patch("dgilit.pmc_pipeline.time.sleep"), self.assertRaisesRegex(ValueError, 'expected 2 IDs, received 1'):
            p.discover()
        self.assertFalse(p.status()["discovery_complete"])
        self.assertEqual(p.store.db.execute("SELECT status FROM partitions").fetchone()[0], "failed")

    def test_live_count_changes_do_not_require_a_second_search_response(self):
        p = self.pipeline()
        observed_sizes = []
        def search(start, end, offset=0, size=0):
            observed_sizes.append(size)
            # A stale count-only response reports one article; a full response
            # reports two. Discovery must never combine those responses.
            return (2, ['PMC1', 'PMC2']) if size else (1, [])
        p.client.search = search
        p.discover()
        self.assertEqual(observed_sizes, [9000])
        self.assertEqual(p.status()['corpus_articles'], 2)
        self.assertTrue(p.status()['discovery_complete'])

    def test_transient_incomplete_response_is_retried(self):
        p = self.pipeline()
        p.client.search = MagicMock(side_effect=[(2,['PMC1']), (2,['PMC1','PMC2'])])
        with patch("dgilit.pmc_pipeline.time.sleep"):
            p.discover()
        self.assertEqual(p.client.search.call_count,2)
        self.assertEqual(p.status()['corpus_articles'],2)
        self.assertTrue(p.status()['discovery_complete'])

    def test_duplicate_ids_cannot_complete_a_partition(self):
        p = self.pipeline()
        p.client.search = MagicMock(return_value=(2,['PMC1','PMC1']))
        with patch("dgilit.pmc_pipeline.time.sleep"), self.assertRaisesRegex(ValueError, '1 unique'):
            p.discover()
        self.assertEqual(p.client.search.call_count,3)
        self.assertFalse(p.status()['discovery_complete'])

    def test_date_splitting_and_duplicate_ids(self):
        p = self.pipeline()
        ranges = []
        def search(start, end, offset=0, size=0):
            ranges.append((start, end))
            if (end-start).days > 200:
                return 10000, []
            return 1, ["PMC1"] if size else []
        p.client.search = search
        p.discover()
        self.assertTrue(p.status()["discovery_complete"])
        self.assertEqual(p.status()["corpus_articles"], 1)
        self.assertIn((date(2020,1,1), date(2020,12,31)), ranges)
        complete = p.store.db.execute("SELECT start,end FROM partitions WHERE status='complete' ORDER BY start").fetchall()
        self.assertEqual(complete[0][0], "2020-01-01")
        self.assertEqual(complete[-1][1], "2020-12-31")
        p.store.set_meta("discovery_complete",False)
        p.client.search = MagicMock(side_effect=AssertionError('Completed split must not be queried again'))
        p.discover()
        p.client.search.assert_not_called()

    def test_all_contexts_and_aggregation(self):
        llm = FakeLLM([True, RuntimeError("failure")])
        llm.context_windows = lambda *args: [(0, 10), (8, 19)]
        p = self.pipeline(llm=llm)
        p.run()
        manifest = p.export()
        import polars as pl
        pairs = pl.read_parquet(self.root / "exports" / manifest["generation"] / "article_pairs" / "*.parquet")
        self.assertEqual(pairs["positive_contexts"][0], 1)
        self.assertEqual(pairs["unresolved_contexts"][0], 1)
        self.assertEqual(llm.calls, 2)

    def test_single_date_over_esearch_limit_splits_uid_ranges(self):
        p = PMCPipeline(self.root, PipelineConfig(start="2020-01-01", end="2020-01-01"),
                        client=FakeClient({}), min_free_bytes=0)
        import re
        def search(start, end, offset=0, size=0, query=None):
            low, high = 1, 10001
            if query:
                match = re.search(r'(\d+):(\d+)\[uid\]', query)
                low, high = max(low, int(match[1])), min(high, int(match[2]))
            count = max(0,high-low+1)
            if query and ' NOT ' in query:
                return 10001-count, []
            return count, [f"PMC{i}" for i in range(low+offset,min(high+1,low+offset+size))]
        p.client.search = search
        p.discover()
        self.assertEqual(p.status()["corpus_articles"],10001)
        self.assertTrue(p.status()["discovery_complete"])

    def test_ungrounded_candidates_are_preserved_but_not_classified(self):
        p = self.pipeline()
        base = p.pretagger.tag_block
        def ungrounded(*args):
            result = base(*args)
            result.entities[0].concept = NormalizedConcept()
            return result
        p.pretagger.tag_block = ungrounded
        p.run()
        self.assertEqual(p.llm.calls, 0)
        self.assertEqual(p.status()["articles"], {"no_candidates": 1})
        self.assertEqual(p.store.db.execute("SELECT count(*) FROM entities").fetchone()[0], 2)

    def test_cache_only_evicts_completed_and_disk_guard(self):
        p = self.pipeline()
        p.discover()
        p.store.cache_put(article(), 100000)
        p.store.evict_cache(0)
        self.assertIsNotNone(p.store.cache_get("PMC1"))
        p.store.db.execute("UPDATE articles SET status='complete'")
        p.store.evict_cache(0)
        self.assertIsNone(p.store.cache_get("PMC1"))
        with patch("dgilit.pmc_store.shutil.disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaises(ResourceLimit):
                p.store.check_space(100)

    def test_changed_fingerprint_is_rejected(self):
        p = self.pipeline()
        p.run()
        original = p.execution_hash
        del p.execution_hash
        with patch("dgilit.pmc_pipeline._model_revision", return_value="new-revision"):
            with self.assertRaises(ValueError):
                p._ensure_models()
        self.assertNotEqual(original, digest({"different": True}))

    def test_pilot_caps_calls_and_full_phase_reuses_work(self):
        llm = FakeLLM()
        llm.context_windows = lambda *args: [(0, 10), (8, 19)]
        papers = {f"PMC{i}":article(f"PMC{i}") for i in range(1,102)}
        p = self.pipeline(papers,llm)
        p.run(pilot_only=True)
        self.assertEqual(llm.calls,100)
        self.assertEqual(p.status()["pilot"]["sample_articles"],100)
        self.assertEqual(p.status()["pilot"]["sample_context_tasks"],200)
        p.run()
        self.assertEqual(llm.calls,202)
        self.assertTrue(p.status()["successful_completion"])

    def test_concurrent_export_does_not_reset_running_tasks(self):
        p = self.pipeline()
        p.run()
        with p.store.db:
            p.store.db.execute("UPDATE tasks SET status='running'")
        with p.store.writer_lock():
            # Simulate an active inference after writer-lock crash recovery.
            with p.store.db:
                p.store.db.execute("UPDATE tasks SET status='running'")
            reader = PMCPipeline(self.root, min_free_bytes=0)
            reader.export()
            self.assertEqual(p.store.db.execute("SELECT status FROM tasks").fetchone()[0],'running')

    def test_incomplete_preparation_cannot_produce_article_negative(self):
        p=self.pipeline(llm=FakeLLM([False]))
        p.run()
        with p.store.db:
            p.store.db.execute("UPDATE articles SET status='failed',error='Preparation incomplete'")
        manifest=p.export()
        import polars as pl
        pairs=pl.read_parquet(self.root/'exports'/manifest['generation']/'article_pairs'/'*.parquet')
        self.assertEqual(pairs['pair_status'][0],'unresolved')
        self.assertIsNone(pairs['interaction'][0])


class TextAndRetrievalTests(unittest.TestCase):
    def test_qwen_context_windows_preserve_text_and_fit_exact_prompt(self):
        from dgilit.qwen import Qwen3JsonClient
        client = object.__new__(Qwen3JsonClient)
        client._tokenizer = CharacterTokenizer()
        client.max_tokens = 10
        client._context_window = 500
        client.prompt_tokens = lambda system,user,schema: len(user)+30
        prompt = SimpleNamespace(build_payload=lambda text,drug,gene: text,
                                 build_system_prompt=lambda: '', build_user_prompt=lambda text: text)
        text='a'*1000
        windows=client.context_windows(prompt,text,'Drug','GENE',{},256)
        self.assertGreater(len(windows),1)
        self.assertEqual({i for a,b in windows for i in range(a,b)},set(range(len(text))))
        self.assertTrue(all(b-a+30+10<=500 for a,b in windows))
        client._context_window=100
        with self.assertRaises(ValueError):
            client.context_windows(prompt,text,'Drug','GENE',{},256)

    def test_long_ner_offsets_and_no_truncation(self):
        tokenizer = CharacterTokenizer()
        windows = list(text_windows(tokenizer, "abcdefghijklmnop", 6, 2))
        covered = {i for a,b in windows for i in range(a,b)}
        self.assertEqual(covered, set(range(16)))
        tagger = BioBertEntityTagger(TaggerConfig(include_genes=False, long_text_overlap=2))
        pipe = MagicMock()
        pipe.tokenizer = tokenizer
        pipe.model.config.max_position_embeddings = 8
        def tag(texts, **kwargs):
            return [[{"word": c, "start": i, "end": i+1, "score": 1.0, "entity_group": "CHEMICAL"}
                     for i,c in enumerate(text) if c == "j"] for text in texts]
        pipe.side_effect = tag
        tagger._pipelines["drug"] = pipe
        mentions = tagger.tag_text("abcdefghijklmnop")
        self.assertEqual([(m.text,m.start,m.end) for m in mentions], [("j",9,10)])

    def test_nested_sections_without_pmid_and_namespace(self):
        xml = '''<article xmlns="urn:jats"><front><article-meta><article-id pub-id-type="pmc">77</article-id>
        <abstract><p>Abstract.</p></abstract><permissions><license>CC BY</license></permissions></article-meta></front>
        <body><sec id="r"><title>Results</title><p>Parent.</p><sec id="r1"><title>Detail</title><p>Child.</p></sec></sec></body></article>'''
        a = parse_pmc_article(ET.fromstring(xml))
        self.assertIsNone(a.pmid)
        self.assertTrue(a.full_text_available)
        self.assertEqual(a.license, "CC BY")
        self.assertEqual([s.text for s in a.sections], ["Abstract.","Parent.","Child."])
        self.assertEqual(a.sections[2].parent_label, "RESULTS")
        abstract = xml[:xml.index('<body>')] + '</article>'
        self.assertFalse(parse_pmc_article(ET.fromstring(abstract)).full_text_available)

    def test_partial_batch_and_abstract_fallback(self):
        client = PMCClient(transport=MagicMock())
        def fetch(ids):
            if len(ids)>1:
                raise RuntimeError("batch failed")
            return {"PMC1": article()} if ids==["PMC1"] else {}
        client._fetch = fetch
        client.summaries = lambda ids: {"PMC2": {"title":"Fallback", "articleids":[{"idtype":"pmid","value":"42"}]}}
        client.transport.request.return_value = b'<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>42</PMID><Article><Abstract><AbstractText>Fallback abstract</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>'
        found = client.fetch_articles(["PMC1","PMC2"])
        self.assertEqual(found["PMC2"].coverage,"abstract_only")
        self.assertEqual(found["PMC2"].sections[0].source,"pubmed")
        self.assertEqual(found["PMC1"].coverage,"full_text")

    def test_transient_http_retry_and_retry_after(self):
        transport = NcbiTransport(retries=1,limiter=MagicMock())
        error = urllib.error.HTTPError("url",429,"slow",{"Retry-After":"2"},None)
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'ok'
        with patch("urllib.request.urlopen",side_effect=[error,response]), patch("dgilit.pmc.time.sleep") as sleep:
            self.assertEqual(transport.request("test",db="pmc"),b'ok')
            sleep.assert_called_with(2)
        self.assertEqual(transport.limiter.wait.call_count,2)

    def test_normalizer_does_not_cache_transport_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = SQLiteNormalizerCache(Path(tmp)/'cache.sqlite')
            cache.set("gene","GENE",{"error_message":"old failure"})
            normalizer = ViccNormalizer(cache=cache)
            response = MagicMock()
            response.json.return_value = {"match_type": 100, "gene": {"id":"gene:1", "label":"GENE"}}
            with patch("dgilit.tagging.normalizers.requests.get", return_value=response) as request:
                self.assertTrue(normalizer.normalize("gene","GENE").is_grounded)
                self.assertTrue(normalizer.normalize("gene","GENE").is_grounded)
                self.assertEqual(request.call_count,1)
            with patch("dgilit.tagging.normalizers.requests.get",side_effect=RuntimeError("offline")):
                self.assertIsNotNone(normalizer.normalize("gene","OTHER").error_message)
                self.assertIsNone(cache.get("gene","OTHER"))


if __name__ == '__main__':
    unittest.main()
