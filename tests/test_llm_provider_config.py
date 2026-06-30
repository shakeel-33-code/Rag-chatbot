import os
import sys
import unittest
from unittest.mock import patch


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import config


class LlmProviderConfigTests(unittest.TestCase):
    def test_huggingface_provider_resolution(self):
        with patch.object(config, "_env_values", return_value={"LLM_PROVIDER": "huggingface"}):
            self.assertEqual("huggingface", config.get_llm_provider())
            self.assertIn("router.huggingface.co", config.get_llm_base_url())

    def test_nvidea_alias_maps_to_nvidia(self):
        with patch.object(config, "_env_values", return_value={"LLM_PROVIDER": "nvidea"}):
            self.assertEqual("nvidia", config.get_llm_provider())

    def test_nvidia_provider_model_and_base_url(self):
        with patch.object(
            config,
            "_env_values",
            return_value={"LLM_PROVIDER": "nvidia"},
        ):
            self.assertEqual("nvidia", config.get_llm_provider())
            self.assertEqual("meta/llama-3.1-8b-instruct", config.get_llm_model())
            self.assertIn("integrate.api.nvidia.com", config.get_llm_base_url())

    def test_ollama_provider_model_and_base_url(self):
        with patch.object(config, "_env_values", return_value={"LLM_PROVIDER": "ollama"}):
            self.assertEqual("ollama", config.get_llm_provider())
            self.assertEqual("gemma4:31b", config.get_llm_model())
            self.assertEqual("https://ollama.com/v1", config.get_llm_base_url())

    def test_invalid_provider_raises(self):
        with patch.object(config, "_env_values", return_value={"LLM_PROVIDER": "invalid-provider"}):
            with self.assertRaises(ValueError):
                config.get_llm_provider()


if __name__ == "__main__":
    unittest.main()
