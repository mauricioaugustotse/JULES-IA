import unittest
from unittest.mock import MagicMock, patch
import json
import sys
import os

# Add parent directory to path to import the script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import SESSOES_TSE_noticias_viaAPI as script

class TestSessoesTSENoticias(unittest.TestCase):

    def test_extract_json_simple(self):
        text = '{"key": "value"}'
        result = script._extract_json(text)
        self.assertEqual(result, {"key": "value"})

    def test_extract_json_markdown(self):
        text = '```json\n{"key": "value"}\n```'
        result = script._extract_json(text)
        self.assertEqual(result, {"key": "value"})

    def test_extract_json_garbage_around(self):
        text = 'Here is the json: {"key": "value"} thanks.'
        result = script._extract_json(text)
        self.assertEqual(result, {"key": "value"})

    def test_normalize_url(self):
        self.assertEqual(script._normalize_url("example.com"), "https://example.com")
        self.assertEqual(script._normalize_url("https://example.com/"), "https://example.com/")
        self.assertEqual(script._normalize_url(' "example.com" '), "https://example.com")

    def test_classify_urls(self):
        urls = [
            "https://tse.jus.br/news",
            "https://tre-sp.jus.br/news",
            "https://folha.uol.com.br/politica",
            "https://unknown.com"
        ]
        tse, tre, geral = script._classify_urls(urls)
        self.assertIn("https://tse.jus.br/news", tse)
        self.assertIn("https://tre-sp.jus.br/news", tre)
        self.assertIn("https://folha.uol.com.br/politica", geral)
        self.assertNotIn("https://unknown.com", tse)
        self.assertNotIn("https://unknown.com", tre)
        self.assertNotIn("https://unknown.com", geral)

    @patch('SESSOES_TSE_noticias_viaAPI.genai.Client')
    def test_call_gemini_mock(self, mock_client_cls):
        # Mock the client instance and method
        mock_client = mock_client_cls.return_value
        mock_response = MagicMock()
        mock_response.text = '{"noticia_TSE": [], "noticia_TRE": [], "noticia_geral": []}'
        mock_client.models.generate_content.return_value = mock_response

        # Call the function (we need to bypass the real import check if possible, or just mock it)
        # Since we imported script, script.genai.Client is mocked

        # We need to test _call_gemini_with_web_search.
        # Note: The script imports types from google.genai, which might not be installed in the environment running this test if it's strictly isolated.
        # However, for this step, I'm assuming I can mock the objects.

        try:
            from google.genai import types
        except ImportError:
            # If types is not available, we can't fully test this function without mocking types too.
            # But the script handles ImportError for google.genai by exiting.
            # So if we are here, either it is installed or we mocked it in sys.modules?
            # The script does `try: from google.genai ...`.
            pass

        # If google.genai is not present, the script might have exited or raised SystemExit on import if it was main execution,
        # but since we imported it as a module, the import block ran.
        # If the import failed, script.genai might be undefined.

        if not hasattr(script, 'genai'):
             print("Skipping API test because google-genai is not installed")
             return

        result = script._call_gemini_with_web_search(
            client=mock_client,
            model="gemini-test",
            prompt="test",
            max_retries=1,
            search_tools=None
        )
        self.assertEqual(result, mock_response.text)

    @patch('SESSOES_TSE_noticias_viaAPI.genai.Client')
    def test_call_gemini_retry(self, mock_client_cls):
        if not hasattr(script, 'genai') or not hasattr(script, 'errors'):
             return

        mock_client = mock_client_cls.return_value
        mock_response = MagicMock()
        mock_response.text = '{"success": true}'

        from google.genai import errors

        # Make side_effect raise error then return success
        # Mocking ClientError is tricky because it might require arguments, let's try just instantiating it or a subclass
        # Actually, let's just mock the call to raise it.
        # We assume errors.ClientError is available since we imported it.

        # Note: ClientError in google.genai might need args.
        # Using a simple Exception that mimics it might not work if the catch block checks type strictly.
        # But we are catching real errors.ClientError.

        # Let's try to instantiate it. If it fails, we will skip this specific check or use a generic Exception if checking generic handling.
        # But our code catches (errors.ClientError, errors.ServerError).

        try:
            err = errors.ClientError("Rate limit", {})
        except Exception:
            # Fallback for test if signature is complex
            err = errors.ClientError("Rate limit", {}, None)

        mock_client.models.generate_content.side_effect = [err, mock_response]

        with patch('time.sleep') as mock_sleep: # speed up test
            result = script._call_gemini_with_web_search(
                client=mock_client,
                model="gemini-test",
                prompt="test",
                max_retries=2,
                search_tools=None
            )

        self.assertEqual(result, '{"success": true}')
        self.assertEqual(mock_client.models.generate_content.call_count, 2)

    @patch('SESSOES_TSE_noticias_viaAPI.genai.Client')
    @patch('time.sleep')
    def test_429_retry_logic(self, mock_sleep, mock_client_cls):
        if not hasattr(script, 'genai') or not hasattr(script, 'errors'):
             return

        from google.genai import errors

        mock_client = mock_client_cls.return_value

        # Create a 429 error
        # We assume errors.ClientError is available
        err_429 = errors.ClientError(code=429, response_json={})

        mock_response = MagicMock()
        mock_response.text = '{"success": true}'

        # Fail once with 429, then succeed
        mock_client.models.generate_content.side_effect = [err_429, mock_response]

        result = script._call_gemini_with_web_search(
            client=mock_client,
            model="gemini-test",
            prompt="test",
            max_retries=3,
            search_tools=None
        )

        self.assertEqual(result, '{"success": true}')
        # Check that sleep(60) was called
        mock_sleep.assert_any_call(60)

    @patch('SESSOES_TSE_noticias_viaAPI.genai.Client')
    @patch('time.sleep')
    def test_403_fatal_logic(self, mock_sleep, mock_client_cls):
        if not hasattr(script, 'genai') or not hasattr(script, 'errors'):
             return

        from google.genai import errors

        mock_client = mock_client_cls.return_value

        # Create a 403 error
        err_403 = errors.ClientError(code=403, response_json={})

        mock_client.models.generate_content.side_effect = err_403

        with self.assertRaises(errors.ClientError):
            script._call_gemini_with_web_search(
                client=mock_client,
                model="gemini-test",
                prompt="test",
                max_retries=3,
                search_tools=None
            )

        # Should verify no retries happened (call count 1)
        self.assertEqual(mock_client.models.generate_content.call_count, 1)

if __name__ == '__main__':
    unittest.main()
