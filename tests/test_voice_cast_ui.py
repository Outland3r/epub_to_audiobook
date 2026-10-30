import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr
from pydub import AudioSegment

from audiobook_generator.ui.voice_cast_ui import (
    analyze_chapter,
    load_plan,
    plan_tables,
    render_chapter,
    reviewed_plan,
    save_plan,
)


def sample_plan():
    return {
        "version": 1,
        "title": "Chapter",
        "book_title": "Book",
        "author": "Author",
        "chapter": 1,
        "voices": {"narrator": "af_bella", "Alice": ""},
        "segments": [
            {"text": "Alice entered.", "speaker": "narrator", "delivery": ""},
            {"text": '"Hello."', "speaker": "Alice", "delivery": "cheerful"},
        ],
    }


class TestVoiceCastUi(unittest.TestCase):
    def test_analysis_populates_review_tables_and_uses_configured_model(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
             patch("audiobook_generator.ui.voice_cast_ui.OpenAI") as client, \
             patch("audiobook_generator.ui.voice_cast_ui.create_plan", return_value=sample_plan()) as create:
            plan, voices, segments, status = analyze_chapter(
                "/tmp/book.epub", 1, "provider/model", "af_bella", ["af_bella", "af_sarah"]
            )
        self.assertEqual(voices, [["narrator", "af_bella"], ["Alice", "af_sarah"]])
        self.assertEqual(segments[1][2], "Alice")
        self.assertIn("Chapter 1", status)
        client.assert_called_once_with(base_url="https://openrouter.ai/api/v1", api_key="test-key")
        self.assertEqual(create.call_args.args[3:], ("provider/model", "af_bella"))
        self.assertEqual(plan["voices"]["Alice"], "")

    def test_analysis_reports_missing_inputs(self):
        with self.assertRaises(gr.Error):
            analyze_chapter(None, 1, "model", "voice", [])
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}), self.assertRaises(gr.Error):
            analyze_chapter("/tmp/book.epub", 1, "model", "voice", [])

    def test_review_does_not_change_original_or_allow_text_loss(self):
        plan = sample_plan()
        voices, segments = plan_tables(plan, ["af_bella", "af_sarah"])
        segments[1][2] = "unknown"
        result = reviewed_plan(plan, voices, segments)
        self.assertEqual(result["segments"][1]["speaker"], "unknown")
        self.assertEqual(plan["segments"][1]["speaker"], "Alice")
        segments[1][1] = "Changed."
        with self.assertRaisesRegex(ValueError, "text was changed"):
            reviewed_plan(plan, voices, segments)
        with self.assertRaisesRegex(ValueError, "All original segments"):
            reviewed_plan(plan, voices, segments[:1])
        with self.assertRaisesRegex(ValueError, "Duplicate speaker"):
            reviewed_plan(plan, voices + [voices[0]], segments[:1])

    def test_save_reload_and_render_reviewed_plan(self):
        plan = sample_plan()
        voices, segments = plan_tables(plan, ["af_bella", "af_sarah"])
        with tempfile.TemporaryDirectory() as directory:
            path, _ = save_plan(plan, voices, segments, directory, "plan.json")
            loaded, loaded_voices, loaded_segments, _ = load_plan(path)
            self.assertEqual(loaded["voices"]["Alice"], "af_sarah")
            self.assertEqual(loaded_segments, segments)
            self.assertEqual(loaded_voices, voices)
            with self.assertRaises(gr.Error):
                save_plan(plan, voices, segments, directory, "plan.json")
            with self.assertRaises(gr.Error):
                save_plan(plan, voices, segments, directory, "../plan.json")
            with patch.dict(os.environ, {"OPENAI_BASE_URL": "http://kokoro/v1"}), \
                 patch("audiobook_generator.ui.voice_cast_ui.OpenAI"), \
                 patch("audiobook_generator.ui.voice_cast_ui.render_plan") as render:
                audio, preview, _ = render_chapter(loaded, loaded_voices, loaded_segments, directory, "chapter.mp3")
            self.assertEqual(audio, preview)
            self.assertEqual(render.call_args.args[0]["voices"]["Alice"], "af_sarah")
            self.assertEqual(render.call_args.args[1], Path(directory) / "chapter.mp3")

    def test_bad_plan_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps({"version": 1}), encoding="utf-8")
            with self.assertRaises(gr.Error):
                load_plan(str(path))

    @unittest.skipUnless(os.environ.get("KOKORO_INTEGRATION") == "1", "requires a running Kokoro server")
    def test_ui_render_uses_live_kokoro(self):
        plan = sample_plan()
        voices, segments = plan_tables(plan, ["af_bella", "af_sarah"])
        with tempfile.TemporaryDirectory() as directory:
            audio, preview, _ = render_chapter(plan, voices, segments, directory, "chapter.mp3")
            self.assertEqual(audio, preview)
            self.assertGreater(len(AudioSegment.from_mp3(audio)), 1000)

    @unittest.skipUnless(os.environ.get("VOICE_UI_TEST_URL"), "requires a running test WebUI")
    def test_browser_api_load_review_save_and_render(self):
        from gradio_client import Client, handle_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            plan = sample_plan()
            plan["voices"]["Alice"] = "af_sarah"
            path.write_text(json.dumps(plan), encoding="utf-8")
            client = Client(os.environ["VOICE_UI_TEST_URL"], verbose=False)
            voices, segments, status = client.predict(handle_file(str(path)), api_name="/load_plan")
            self.assertIn("Loaded chapter 1", status)
            segments["data"][1][2] = "unknown"
            _, status = client.predict(voices, segments, directory, "reviewed.json", api_name="/save_plan")
            self.assertIn("Plan saved", status)
            saved = json.loads((Path(directory) / "reviewed.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["segments"][1]["speaker"], "unknown")
            audio, preview, status = client.predict(voices, segments, directory, "preview.mp3", api_name="/render_chapter")
            self.assertIn("Rendered chapter", status)
            self.assertGreater(len(AudioSegment.from_mp3(audio)), 1000)
            self.assertGreater(len(AudioSegment.from_mp3(preview)), 1000)


if __name__ == "__main__":
    unittest.main()
