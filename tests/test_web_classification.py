
import unittest
import sys
import os
import re
import importlib.util
from unittest.mock import MagicMock

# Mock dependencies to allow import
sys.modules["dotenv"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.genai"] = MagicMock()
sys.modules["google.genai.types"] = MagicMock()
sys.modules["google.genai.errors"] = MagicMock()

# Load the module dynamically due to special characters in filename
module_name = "SESSOES_TSE_notícias_WEB"
# Assuming test is run from repo root
file_path = "SESSOES_TSE_notícias_WEB.py"
if not os.path.exists(file_path):
    # Try finding it relative to test file if run from tests dir
    file_path = "../SESSOES_TSE_notícias_WEB.py"

if os.path.exists(file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass
else:
    module = None

class TestWebClassification(unittest.TestCase):

    def setUp(self):
        if module is None:
            self.skipTest("Module SESSOES_TSE_notícias_WEB.py not found")

    def test_normalize_url(self):
        norm = module._normalize_url
        self.assertEqual(norm("example.com"), "https://example.com")
        self.assertEqual(norm("http://example.com"), "http://example.com")
        self.assertEqual(norm("  example.com. "), "https://example.com")
        self.assertEqual(norm(""), "")

    def test_domain_from_url(self):
        dom = module._domain_from_url
        self.assertEqual(dom("https://example.com"), "example.com")
        self.assertEqual(dom("https://www.example.com/foo"), "example.com")
        self.assertEqual(dom("http://sub.domain.com"), "sub.domain.com")
        self.assertEqual(dom("https://example.com:8080"), "example.com")
        # Edge cases
        self.assertEqual(dom("https://user:pass@example.com/"), "example.com")

    def test_classify_urls(self):
        classify = module._classify_urls
        urls = [
            "tse.jus.br",
            "https://www.tse.jus.br/some/news",
            "tre-sp.jus.br",
            "http://tre-rj.jus.br/decision",
            "folha.uol.com.br",
            "https://g1.globo.com/politica",
            "random.com",
            "jus.br", # ends with jus.br but not matched specifically
        ]

        tse, tre, geral = classify(urls)

        self.assertIn("https://tse.jus.br", tse)
        self.assertIn("https://www.tse.jus.br/some/news", tse)

        self.assertIn("https://tre-sp.jus.br", tre)
        self.assertIn("http://tre-rj.jus.br/decision", tre)

        self.assertIn("https://folha.uol.com.br", geral)
        self.assertIn("https://g1.globo.com/politica", geral)

        # random.com -> "jus.br" not in domain -> geral
        self.assertIn("https://random.com", geral)

        # jus.br -> "jus.br" in domain -> not general?
        # logic: if "jus.br" not in domain: geral.append
        # so "jus.br" domain has "jus.br" in it. So it is NOT appended to geral.
        # And it's not TSE or TRE. So it is dropped.
        # Let's verify normalized version
        norm_jus = "https://jus.br"
        self.assertNotIn(norm_jus, geral)
        self.assertNotIn(norm_jus, tse)
        self.assertNotIn(norm_jus, tre)

if __name__ == '__main__':
    unittest.main()
