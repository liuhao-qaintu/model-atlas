"""来源身份不能互相覆盖；正文需进入真实章节并可完整提供。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.infra.config import CONTENT_ROOT
from backend.generation.evidence_context import select_excerpt
from backend.infra.knowledge import SECTION_NAMES, SECTION_SOURCES, parse_sections, source_records
from backend.infra.recent_knowledge import RECENT_TOPICS


class ContentProvenanceTests(unittest.TestCase):
    def test_source_ids_keep_unrelated_publications_and_implementations_distinct(self):
        sources = source_records()
        by_id = {source['id']: source for source in sources}
        self.assertEqual(len(sources), len(by_id))
        self.assertIn('/pytorch/pytorch/', by_id['S44']['url'])
        self.assertIn('/notes/autograd', by_id['S45']['url'])
        self.assertIn('/google-research/big_vision', by_id['S50']['url'])
        self.assertIn('/facebookresearch/dinov3', by_id['S51']['url'])
        for topic, repository in [('siglip2', 'big_vision'), ('dinov3', 'dinov3')]:
            source_ids = next(t['source_ids'] for t in RECENT_TOPICS if t['id'] == topic)
            self.assertTrue(any(repository in by_id[sid]['url'] for sid in source_ids if sid in by_id))

    def test_duplicate_source_identity_is_rejected_before_seeding(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / '首批资料清单.md').write_text(
                '| S01 | first | [first](https://example.com/a) | scope |\n'
                '| S01 | second | [second](https://example.com/b) | scope |\n', encoding='utf-8')
            with patch('backend.infra.knowledge.CONTENT_ROOT', path):
                with self.assertRaises(ValueError):
                    source_records()

    def test_detr_chapters_have_valid_sources_and_fit_full_generation_context(self):
        text = (CONTENT_ROOT / '模型卡' / '09_DETR.md').read_text(encoding='utf-8')
        sections = {SECTION_NAMES[s['title']]: s for s in parse_sections(text)}
        valid = {s['id'] for s in source_records()}
        for slug in ('overview', 'mechanism', 'io', 'formula', 'code', 'practice', 'figures'):
            with self.subTest(slug=slug):
                self.assertTrue(set(SECTION_SOURCES['detr'][slug]) <= valid)
                excerpt = select_excerpt(sections[slug]['content'], '详细解释')
                self.assertEqual(excerpt['strategy'], 'full')
                self.assertEqual(excerpt['text'], sections[slug]['content'])


if __name__ == '__main__':
    unittest.main()
