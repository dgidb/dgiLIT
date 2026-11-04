import gzip
import json
from pathlib import Path
import sqlite3
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest

from api.app import create_app
from api.config import Settings
from api.provenance import digest
from api.repository import Repository, DatabaseUnavailable
from api.sources import SourceService, SourceBusy
from tests.fixtures import populate, TASK, TEXT


@pytest.fixture
def setup(tmp_path):
    data=populate(tmp_path/"run")
    prompts=tmp_path/"prompts";prompts.mkdir();(prompts/"v2.text").write_text("test prompt")
    settings=Settings(data,prompts,summary_ttl=30)
    provider=MagicMock()
    provider.fetch.return_value=([{"text":TEXT}],[])
    repo=Repository(settings)
    return settings,repo,provider,TestClient(create_app(settings,repo,provider))


def test_full_cohort_and_pagination(setup):
    _,_,_,client=setup
    summary=client.get('/api/summary').json()
    assert summary['positive_contexts']==62
    assert summary['supporting_articles']==2
    assert {x['cohort']:x['positive_contexts'] for x in summary['cohorts']}=={'pilot':52,'beyond_pilot':10}
    first=client.get('/api/contexts').json();second=client.get('/api/contexts?page=2').json()
    assert first['total']==second['total']==62
    assert len(first['items'])==50 and len(second['items'])==12
    assert not ({r['task_id'] for r in first['items']} & {r['task_id'] for r in second['items']})
    assert client.get('/api/contexts?cohort=beyond_pilot').json()['total']==10
    assert client.get('/api/contexts?cohort=pilot').json()['total']==52


def test_filters_validation_and_binding(setup):
    client=setup[3]
    assert client.get('/api/contexts?drug=drug:1&gene=bcl2&article=PMC2&coverage=abstract_only').json()['total']==10
    assert client.get('/api/contexts?article=123').json()['total']==52
    assert client.get('/api/contexts',params={'gene':"' OR 1=1 --"}).json()['total']==0
    for suffix in ['page=0','page_size=1001','cohort=other','coverage=other']:
        assert client.get('/api/contexts?'+suffix).status_code==422
    assert client.get('/api/contexts?page=999').json()['items']==[]


def test_detail_and_history_provenance(setup):
    settings,_,_,client=setup
    value=client.get('/api/contexts/'+TASK(1)).json()
    assert value['entities'][0]['concept']['normalizer']=='vicc:therapy'
    assert value['prompt']['status']=='verified' and value['prompt']['template']=='test prompt'
    assert value['metadata']['execution_fingerprint']['model_revisions']['test-model']=='revision'
    assert value['task']['result']['generation_stats']['prompt_tokens']==100
    assert value['related_contexts'][0]['contexts']==52
    historical=client.get('/api/contexts/'+TASK(63)).json()
    assert historical['task']['is_current'] is False
    (settings.prompt_dir/'v2.text').write_text('changed prompt')
    value=client.get('/api/contexts/'+TASK(1)).json()
    assert value['prompt']['status']=='hash_mismatch' and value['prompt']['template'] is None
    assert client.get('/api/contexts/'+TASK(999)).status_code==404
    assert client.get('/api/contexts/invalid').status_code==404


def test_source_cached_and_network_retrieval_without_writes(setup):
    settings,_,provider,client=setup
    database=settings.data_dir/'state.sqlite'
    before=database.read_bytes()
    value=client.get('/api/contexts/'+TASK(1)+'/source').json()
    assert value['state']=='verified' and value['origin']=='cli_cache' and value['context']==TEXT
    provider.fetch.assert_not_called()
    value=client.get('/api/contexts/'+TASK(53)+'/source').json()
    assert value['state']=='unavailable'
    value=client.post('/api/contexts/'+TASK(53)+'/source').json()
    assert value['state']=='verified' and value['origin']=='pmc_retrieval'
    assert provider.fetch.call_count==1
    client.post('/api/contexts/'+TASK(53)+'/source')
    assert provider.fetch.call_count==1 # bounded memory cache
    assert not (settings.data_dir/'text_cache/PMC2.json.gz').exists()
    assert database.read_bytes()==before


@pytest.mark.parametrize('sections,errors,state',[
    ([{'text':'changed source'}],[],'hash_mismatch'),([],[],'unavailable'),([],['offline'],'retrieval_error')])
def test_retrieval_outcomes(setup,sections,errors,state):
    _,_,provider,client=setup
    provider.fetch.return_value=(sections,errors)
    value=client.post('/api/contexts/'+TASK(53)+'/source').json()
    assert value['state']==state and value['context'] is None


def test_provider_exception_and_busy(setup):
    settings,repo,provider,client=setup
    provider.fetch.side_effect=RuntimeError('private upstream error')
    response=client.post('/api/contexts/'+TASK(53)+'/source')
    assert response.json()['state']=='retrieval_error' and 'private' not in response.text
    service=client.app.state.sources
    service._fetch_lock.acquire()
    try: assert client.post('/api/contexts/'+TASK(53)+'/source').status_code==429
    finally: service._fetch_lock.release()
    assert client.post('/api/contexts/'+TASK(999)+'/source').status_code==404


def test_hash_verification_requires_section_and_context(setup):
    settings,repo,_,_=setup
    service=SourceService(settings)
    task=repo.detail(TASK(1))['task']
    task['context_hash']='wrong'
    assert service.cached(task)['state']=='hash_mismatch'
    task['context_hash']=digest(TEXT);task['section_hash']='wrong'
    assert service.cached(task)['state']=='hash_mismatch'
    with gzip.open(settings.data_dir/'text_cache/PMC1.json.gz','wt') as h:h.write('not json')
    assert service.cached(task)['state']=='retrieval_error'


def test_read_only_sqlite_and_missing_database(setup,tmp_path):
    settings,repo,_,client=setup
    with pytest.raises(DatabaseUnavailable):
        with repo.connection() as db:
            assert db.execute('PRAGMA query_only').fetchone()[0]==1
            db.execute("UPDATE meta SET value=value")
    missing=tmp_path/'missing'
    client=TestClient(create_app(Settings(missing)))
    assert client.get('/api/health').status_code==503
    assert not missing.exists()


def test_summary_cache_and_manual_refresh(setup):
    _,repo,_,client=setup
    with patch.object(repo,'summary',wraps=repo.summary) as method:
        client.get('/api/summary');client.get('/api/summary')
        assert method.call_count==1
        client.get('/api/summary?refresh=true')
        assert method.call_count==2


def test_empty_results(setup):
    settings,_,_,client=setup
    with sqlite3.connect(settings.data_dir/'state.sqlite') as db:db.execute('DELETE FROM tasks')
    assert client.get('/api/contexts').json()['total']==0
    assert client.get('/api/summary').json()['positive_contexts']==0


def test_pubmed_fallback_uses_existing_parser_and_no_models(setup):
    from api.upstream import PMCSourceProvider
    settings,repo,_,_=setup
    task=repo.detail(TASK(53))['task'];task['pmid']='123';task['section_metadata']['source']='pubmed'
    xml=b'<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article><Abstract><AbstractText>PubMed abstract</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>'
    with patch('dgilit.pmc.PMCClient') as client,patch('dgilit.pmc.NcbiTransport') as transport:
        client.return_value.fetch_articles.return_value={}
        client.return_value.last_errors={}
        transport.return_value.request.return_value=xml
        sections,errors=PMCSourceProvider().fetch(task)
    assert sections[0]['text']=='PubMed abstract' and sections[0]['source']=='pubmed'
    assert errors==[]
