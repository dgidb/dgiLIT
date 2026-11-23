"""Small deterministic checkpoints; never populated by the extractor."""
import gzip
import json
import sqlite3
from api.provenance import digest

TASK = lambda i: f"{i:064x}"
TEXT = "Venetoclax inhibits BCL2 and induces apoptosis."


def populate(directory):
    directory.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(directory / "state.sqlite")
    db.executescript('''
    CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT);
    CREATE TABLE articles(pmcid TEXT PRIMARY KEY,pmid TEXT,title TEXT,publication_year TEXT,license TEXT,pilot INTEGER,status TEXT,coverage TEXT,text_hash TEXT);
    CREATE TABLE sections(id TEXT PRIMARY KEY,pmcid TEXT,article_hash TEXT,section_index INTEGER,text_hash TEXT,metadata TEXT);
    CREATE TABLE entities(id TEXT,section_id TEXT,pmcid TEXT,data TEXT);
    CREATE TABLE tasks(id TEXT PRIMARY KEY,pmcid TEXT,article_hash TEXT,section_id TEXT,section_index INTEGER,
        context_start INTEGER,context_end INTEGER,context_hash TEXT,drug_id TEXT,gene_id TEXT,drug TEXT,gene TEXT,
        config_hash TEXT,status TEXT,result TEXT,error TEXT,attempts INTEGER,elapsed_seconds REAL,updated_at TEXT);
    ''')
    db.executemany("INSERT INTO articles VALUES (?,?,?,?,?,?,?,?,?)",[
        ("PMC1","123","A pilot study of apoptosis","2020","CC BY",1,"prepared","full_text","current"),
        ("PMC2",None,"BCL2 in non-pilot research","2021",None,0,"complete","abstract_only","current")])
    section = {"section_type":"results","label":"RESULTS","source":"pmc","section_id":"s1","parent_label":"Evidence"}
    for i in [1,2]:
        db.execute("INSERT INTO sections VALUES (?,?,?,?,?,?)",(f"s{i}",f"PMC{i}","current",0,digest(TEXT),json.dumps(section)))
    entity = {"text":"Venetoclax","entity_type":"drug","start":0,"end":10,"score":.98,"source":"biobert:test",
              "concept":{"concept_id":"drug:1","concept_label":"Venetoclax","normalizer":"vicc:therapy","match_type":100}}
    db.execute("INSERT INTO entities VALUES (?,?,?,?)",("entity","s1","PMC1",json.dumps(entity)))
    metadata = {"config":{"model_id":"test-model","start":"2016-11-03","end":"2026-11-03"},
        "execution_fingerprint":{"prompt_hash":digest("test prompt"),"model_revisions":{"test-model":"revision"},"packages":{"model-library":"1.0"}},
        "discovery_complete":True,"last_run":{"status":"running"}}
    db.executemany("INSERT INTO meta VALUES (?,?)",((k,json.dumps(v)) for k,v in metadata.items()))
    for i in range(1,64):
        pmcid = "PMC1" if i<=52 else "PMC2"
        result = {"drug":"Venetoclax","gene":"BCL2","interaction":True,"evidence":TEXT,"evidence_in_context":True,
            "interaction_type":"inhibition","directionality":"inhibiting","model_id":"test-model","prompt_version":"v2",
            "generation_stats":{"prompt_tokens":100,"generation_tokens":20}}
        db.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(
            TASK(i),pmcid,"old" if i==63 else "current","s1" if pmcid=="PMC1" else "s2",0,0,len(TEXT),digest(TEXT),
            "drug:1","gene:1","Venetoclax","BCL2","config-fingerprint","done",json.dumps(result),None,1,2.5,f"2026-11-04T12:00:{i%60:02}+00:00"))
    db.commit()
    db.close()
    cache = directory / "text_cache"
    cache.mkdir()
    with gzip.open(cache / "PMC1.json.gz", "wt") as handle:
        json.dump({"sections":[{"text":TEXT,**section}]},handle)
    return directory
