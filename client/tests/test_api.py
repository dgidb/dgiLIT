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
    settings=Settings(data,prompts,summary_ttl=30,review_db=tmp_path/"reviews.sqlite")
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


def test_rating_cycle_persists_and_preserves_extraction_database(setup):
    settings,repo,_,client=setup
    original=(settings.data_dir/'state.sqlite').read_bytes()
    assert client.get('/api/contexts/'+TASK(1)).json()['rating'] is None
    assert not settings.review_db.exists()
    saved=client.put('/api/contexts/'+TASK(1)+'/rating',json={'rating':'up'})
    assert saved.status_code==200 and saved.json()['rating']=='up' and saved.json()['rated_at']
    assert settings.review_db.exists()
    assert client.get('/api/contexts/'+TASK(1)).json()['rating']=='up'
    page=client.get('/api/contexts?page_size=100').json()
    assert next(row for row in page['items'] if row['task_id']==TASK(1))['rating']=='up'
    assert all(row['rating'] is None for row in page['items'] if row['task_id']!=TASK(1))
    assert client.put('/api/contexts/'+TASK(1)+'/rating',json={'rating':'down'}).json()['rating']=='down'
    assert client.get('/api/contexts/'+TASK(1)).json()['rating']=='down'
    assert client.put('/api/contexts/'+TASK(1)+'/rating',json={'rating':None}).json()=={'rating':None,'rated_at':None}
    assert client.get('/api/contexts/'+TASK(1)).json()['rating'] is None
    assert (settings.data_dir/'state.sqlite').read_bytes()==original
    with repo.connection() as db:
        assert db.execute("SELECT 1 FROM sqlite_master WHERE name='claim_ratings'").fetchone() is None
    with sqlite3.connect(settings.review_db) as db:
        assert db.execute('SELECT count(*) FROM claim_ratings').fetchone()[0]==0


def test_ratings_shared_across_app_restarts_and_browsers(setup):
    settings,_,_,client=setup
    client.put('/api/contexts/'+TASK(1)+'/rating',json={'rating':'up'})
    another=TestClient(create_app(settings))
    assert another.get('/api/contexts/'+TASK(1)).json()['rating']=='up'
    another.put('/api/contexts/'+TASK(1)+'/rating',json={'rating':'down'})
    assert client.get('/api/contexts/'+TASK(1)).json()['rating']=='down'


def test_ratings_allow_history_and_do_not_transfer_between_tasks(setup):
    _,_,_,client=setup
    response=client.put('/api/contexts/'+TASK(63)+'/rating',json={'rating':'down'})
    assert response.status_code==200
    record=client.get('/api/contexts/'+TASK(63)).json()
    assert record['rating']=='down' and record['task']['is_current'] is False
    assert client.get('/api/contexts/'+TASK(53)).json()['rating'] is None


@pytest.mark.parametrize('body',[{'rating':'invalid'}, {'rating':True}, {'rating':1}, {}, {'rating':'up','extra':'not allowed'}])
def test_invalid_rating_input_creates_no_store(setup,body):
    settings,_,_,client=setup
    assert client.put('/api/contexts/'+TASK(1)+'/rating',json=body).status_code==422
    assert not settings.review_db.exists()


def test_missing_or_nonpositive_claim_cannot_be_rated(setup):
    settings,_,_,client=setup
    assert client.put('/api/contexts/'+TASK(999)+'/rating',json={'rating':'up'}).status_code==404
    assert client.put('/api/contexts/invalid/rating',json={'rating':'up'}).status_code==404
    with sqlite3.connect(settings.data_dir/'state.sqlite') as db:
        db.execute('UPDATE tasks SET result=? WHERE id=?',(json.dumps({'interaction':False}),TASK(1)))
    assert client.put('/api/contexts/'+TASK(1)+'/rating',json={'rating':'up'}).status_code==404
    assert not settings.review_db.exists()


def test_save_failure_keeps_previous_assessment(setup):
    from api.ratings import RatingsUnavailable
    _,_,_,client=setup
    client.put('/api/contexts/'+TASK(1)+'/rating',json={'rating':'up'})
    with patch.object(client.app.state.ratings,'set',side_effect=RatingsUnavailable('Save failed')):
        response=client.put('/api/contexts/'+TASK(1)+'/rating',json={'rating':'down'})
    assert response.status_code==503
    assert client.get('/api/contexts/'+TASK(1)).json()['rating']=='up'


def test_ratings_page_uses_one_batch_lookup(setup):
    _,_,_,client=setup
    store=client.app.state.ratings
    with patch.object(store,'get_many',wraps=store.get_many) as query:
        response=client.get('/api/contexts')
        assert response.status_code==200
        assert query.call_count==1 and len(query.call_args.args[0])==50


def test_review_configuration_cannot_target_extraction_directory(tmp_path,monkeypatch):
    data=tmp_path/'run';data.mkdir()
    for target in [data/'state.sqlite',data/'reviews.sqlite',data/'text_cache/reviews.sqlite']:
        with pytest.raises(ValueError,match='outside the extraction run'):
            Settings(data,review_db=target)
    alias=tmp_path/'alias';alias.symlink_to(data,target_is_directory=True)
    with pytest.raises(ValueError):Settings(data,review_db=alias/'reviews.sqlite')
    monkeypatch.setenv('DGILIT_DATA_DIR',str(data))
    monkeypatch.setenv('DGILIT_REVIEW_DB',str(tmp_path/'app-data/reviews.sqlite'))
    assert Settings.from_environment().review_db==tmp_path/'app-data/reviews.sqlite'


def test_review_filesystem_failure_returns_clear_error(setup,tmp_path):
    settings,_,_,_=setup
    not_directory=tmp_path/'file';not_directory.write_text('file')
    blocked=Settings(settings.data_dir,settings.prompt_dir,review_db=not_directory/'reviews.sqlite')
    client=TestClient(create_app(blocked))
    response=client.put('/api/contexts/'+TASK(1)+'/rating',json={'rating':'up'})
    assert response.status_code==503 and response.json()['code']=='ratings_unavailable'


def test_rating_filters_with_no_review_store(setup):
    settings,_,_,client=setup
    assert client.get('/api/contexts?rating=up').json()['total']==0
    assert client.get('/api/contexts?rating=hide_down').json()['total']==62
    assert not settings.review_db.exists()
    assert client.get('/api/contexts?rating=invalid').status_code==422


def test_rating_filters_include_unrated_when_hiding_down_and_combine_with_search(setup):
    settings,_,_,client=setup
    original=(settings.data_dir/'state.sqlite').read_bytes()
    for task,rating in [(1,'up'),(53,'up'),(2,'down'),(54,'down'),(63,'up')]:
        assert client.put('/api/contexts/'+TASK(task)+'/rating',json={'rating':rating}).status_code==200
    approved=client.get('/api/contexts?rating=up').json()
    assert approved['total']==2 and {r['task_id'] for r in approved['items']}=={TASK(1),TASK(53)}
    hidden=client.get('/api/contexts?rating=hide_down').json()
    assert hidden['total']==60
    assert all(r['rating']!='down' for r in hidden['items'])
    assert any(r['rating'] is None for r in hidden['items'])
    combined=client.get('/api/contexts',params={'rating':'up','cohort':'beyond_pilot','coverage':'abstract_only','article':'PMC2','gene':'bcl2','drug':'drug:1'}).json()
    assert combined['total']==1 and combined['items'][0]['task_id']==TASK(53)
    assert client.get('/api/contexts?rating=hide_down&cohort=beyond_pilot').json()['total']==9
    assert (settings.data_dir/'state.sqlite').read_bytes()==original


def test_rating_filter_applies_before_counting_and_pagination(setup):
    _,_,_,client=setup
    for i in range(1,54):client.app.state.ratings.set(TASK(i),'up')
    first=client.get('/api/contexts?rating=up&page=1').json()
    second=client.get('/api/contexts?rating=up&page=2').json()
    assert first['total']==second['total']==53
    assert len(first['items'])==50 and len(second['items'])==3
    assert all(r['rating']=='up' for r in [*first['items'],*second['items']])
    assert not ({r['task_id'] for r in first['items']} & {r['task_id'] for r in second['items']})
    for i in range(1,54):client.app.state.ratings.set(TASK(i),'down')
    visible=client.get('/api/contexts?rating=hide_down').json()
    assert visible['total']==9 and len(visible['items'])==9


def test_rating_filters_after_switching_and_clearing(setup):
    _,_,_,client=setup
    client.put('/api/contexts/'+TASK(1)+'/rating',json={'rating':'up'})
    assert client.get('/api/contexts?rating=up').json()['total']==1
    client.put('/api/contexts/'+TASK(1)+'/rating',json={'rating':'down'})
    assert client.get('/api/contexts?rating=up').json()['total']==0
    assert client.get('/api/contexts?rating=hide_down').json()['total']==61
    client.put('/api/contexts/'+TASK(1)+'/rating',json={'rating':None})
    assert client.get('/api/contexts?rating=hide_down').json()['total']==62


def test_filter_connections_keep_both_databases_read_only(setup):
    settings,_,_,client=setup
    client.put('/api/contexts/'+TASK(1)+'/rating',json={'rating':'up'})
    before=settings.review_db.read_bytes()
    statements=[]
    original_connect=sqlite3.connect
    def traced(*args,**kwargs):
        db=original_connect(*args,**kwargs)
        db.set_trace_callback(statements.append)
        return db
    with patch('api.repository.sqlite3.connect',side_effect=traced):
        assert client.get('/api/contexts?rating=up').status_code==200
        assert client.get('/api/contexts?rating=hide_down').status_code==200
    attachments=[s for s in statements if s.startswith('ATTACH DATABASE')]
    assert len(attachments)==2 and all('?mode=ro' in s for s in attachments)
    assert not any(s.lstrip().upper().startswith(('INSERT','UPDATE','DELETE','CREATE')) for s in statements)
    assert settings.review_db.read_bytes()==before
