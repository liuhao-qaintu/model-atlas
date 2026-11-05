import unittest
from types import SimpleNamespace
from backend.rag.planning import explicit_models


class ModelAliasTests(unittest.TestCase):
    def fixture(self, aliases):
        models = {mid: {'aliases': values} for mid, values in aliases.items()}
        return SimpleNamespace(models=models, identify_models=lambda text: [mid for mid, value in models.items()
            if any(alias.lower() in text.lower() for alias in value['aliases'])])

    def test_user_uppercase_alias_matches_without_empty_sort_crash(self):
        active = self.fixture({'paper': ['DAE', 'Denoising Autoencoder', '去噪 AE']})
        self.assertEqual(explicit_models('为什么去噪 AE 的目标是干净输入？', active), ['paper'])
        self.assertEqual(explicit_models('DAE 输入是什么？', active), ['paper'])

    def test_nested_model_detection_is_case_insensitive(self):
        active = self.fixture({'clip': ['CLIP'], 'vit': ['ViT']})
        self.assertEqual(explicit_models('CLIP 的 ViT 如何工作？', active), ['clip'])
        self.assertEqual(explicit_models('比较 CLIP 和 ViT', active), ['clip', 'vit'])

    def test_mixed_case_aliases_preserve_question_order(self):
        active = self.fixture({'ae': ['Autoencoder'], 'paper': ['DAE']})
        self.assertEqual(explicit_models('比较 DAE 和 Autoencoder', active), ['paper', 'ae'])


if __name__ == '__main__':
    unittest.main()
