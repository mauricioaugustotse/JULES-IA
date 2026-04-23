import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import local_secrets


class TestLocalSecrets(unittest.TestCase):
    def test_loads_known_secret_files_and_aliases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            (base_dir / "CHAVE_SECRETA_API_Mauricio_local.txt").write_text("openai-secret\n", encoding="utf-8")
            (base_dir / "Chave_Notion.txt").write_text("notion-secret\n", encoding="utf-8")
            (base_dir / "Chave_secreta_Perplexity.txt").write_text("pplx-secret\n", encoding="utf-8")
            (base_dir / "Chave_Gemini.txt").write_text("gemini-secret\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                loaded = local_secrets.load_local_secrets(base_dir=base_dir)

                self.assertEqual(loaded["OPENAI_API_KEY"], "CHAVE_SECRETA_API_Mauricio_local.txt")
                self.assertEqual(loaded["NOTION_API_KEY"], "Chave_Notion.txt")
                self.assertEqual(loaded["PERPLEXITY_API_KEY"], "Chave_secreta_Perplexity.txt")
                self.assertEqual(loaded["GEMINI_API_KEY"], "Chave_Gemini.txt")
                self.assertEqual(os.environ["OPENAI_API_KEY"], "openai-secret")
                self.assertEqual(os.environ["NOTION_API_KEY"], "notion-secret")
                self.assertEqual(os.environ["NOTION_TOKEN"], "notion-secret")
                self.assertEqual(os.environ["PERPLEXITY_API_KEY"], "pplx-secret")
                self.assertEqual(os.environ["PPLX_API_KEY"], "pplx-secret")
                self.assertEqual(os.environ["GEMINI_API_KEY"], "gemini-secret")
                self.assertEqual(os.environ["GOOGLE_API_KEY"], "gemini-secret")

    def test_existing_environment_value_has_priority(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            (base_dir / "Chave_Gemini.txt").write_text("file-secret\n", encoding="utf-8")

            with patch.dict(os.environ, {"GEMINI_API_KEY": "env-secret"}, clear=True):
                loaded = local_secrets.load_local_secrets(base_dir=base_dir)

                self.assertEqual(loaded, {})
                self.assertEqual(os.environ["GEMINI_API_KEY"], "env-secret")
                self.assertEqual(os.environ["GOOGLE_API_KEY"], "env-secret")

    def test_get_secret_returns_first_available_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            (base_dir / "Chave_Gemini.txt").write_text("gemini-secret\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                value = local_secrets.get_secret(
                    "OPENAI_API_KEY",
                    "GEMINI_API_KEY",
                    base_dir=base_dir,
                )

                self.assertEqual(value, "gemini-secret")


if __name__ == "__main__":
    unittest.main()
