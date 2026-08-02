"""Learning-content audit tests."""

from __future__ import annotations

import pytest

from augmentum.knowledge.lang_pack_builder import build_pack
from augmentum.knowledge.packs import PackManager
from augmentum.learning.content_audit import audit_learning_content

_JMDICT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE JMdict [
<!ENTITY v1 "Ichidan verb">
<!ENTITY vt "transitive verb">
<!ENTITY n "noun">
]>
<JMdict>
<entry>
<ent_seq>1358280</ent_seq>
<k_ele><keb>食べる</keb></k_ele>
<r_ele><reb>たべる</reb></r_ele>
<sense><pos>&v1;</pos><pos>&vt;</pos><gloss>to eat</gloss></sense>
</entry>
<entry>
<ent_seq>1578850</ent_seq>
<k_ele><keb>朝ごはん</keb></k_ele>
<r_ele><reb>あさごはん</reb></r_ele>
<sense><pos>&n;</pos><gloss>breakfast</gloss></sense>
</entry>
</JMdict>
"""
_SENT_TSV = "1\tjpn\t彼は朝ごはんを食べる。\n2\teng\tHe eats breakfast.\n"
_LINK_TSV = "1\t2\n2\t1\n"


@pytest.mark.asyncio
async def test_learning_content_audit_path_only():
    report = await audit_learning_content(lang_codes=["es"], pack_manager=None, include_examples=False)

    assert report["summary"]["languages"] == 1
    assert report["summary"]["path_languages"] == 1
    lang = report["languages"]["es"]
    assert lang["catalog"]["status"] == "available"
    assert lang["path"]["present"] is True
    assert lang["path"]["level_system"] == "cefr"
    assert lang["path"]["vocab_count"] > 0
    assert lang["pack"]["installed"] is False
    assert lang["coverage"]["coverage"] is None
    assert lang["game_material"]["basis"] == "path"
    assert lang["game_material"]["games"]["word_garden"]["ready"] is True


@pytest.mark.asyncio
async def test_learning_content_audit_installed_pack(tmp_path):
    pack_dir = tmp_path / "packs"
    pack_dir.mkdir()
    (tmp_path / "JMdict_e").write_text(_JMDICT_XML, encoding="utf-8")
    (tmp_path / "sentences.tsv").write_text(_SENT_TSV, encoding="utf-8")
    (tmp_path / "links.tsv").write_text(_LINK_TSV, encoding="utf-8")
    build_pack(
        out_path=pack_dir / "ja.augpack",
        lang_code="ja",
        jmdict_xml=tmp_path / "JMdict_e",
        tatoeba_sentences=tmp_path / "sentences.tsv",
        tatoeba_links=tmp_path / "links.tsv",
    )

    pm = PackManager(pack_dir)
    await pm.scan()
    try:
        report = await audit_learning_content(pack_manager=pm, lang_codes=["ja"])
    finally:
        await pm.close()

    lang = report["languages"]["ja"]
    assert lang["pack"]["installed"] is True
    assert lang["pack"]["vocab_count"] == 2
    assert lang["pack"]["sentences"]["translated_easy"] == 1
    assert lang["coverage"]["checked"] == lang["path"]["vocab_count"]
    assert lang["coverage"]["resolved"] >= 1
    assert lang["examples"]["checked"] >= 1
    assert lang["examples"]["with_example"] >= 1
    assert lang["game_material"]["basis"] == "pack"
    assert lang["game_material"]["games"]["word_garden"]["ready"] is True
    assert lang["game_material"]["games"]["bubble_pop"]["ready"] is False
    assert any(f["code"] == "low_path_pack_coverage" for f in lang["findings"])
