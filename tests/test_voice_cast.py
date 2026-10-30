import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from ebooklib import epub
from mutagen.id3 import ID3
from pydub import AudioSegment
from pydub.generators import Sine

from audiobook_generator.core.voice_cast import (
    MAX_ANALYSIS_CHARS,
    analysis_batches,
    create_plan,
    identify_speakers,
    render_plan,
    split_dialogue,
    validate_plan,
)


def analysis_client(content):
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(content)))]
    )
    return client


class TestVoiceCast(unittest.TestCase):
    def test_segmentation_preserves_text(self):
        text = 'Mary said, “Hello!” Bob replied, "Hi." Then they left.'
        segments = split_dialogue(text)
        self.assertEqual("".join(segment["text"] for segment in segments), text)
        self.assertEqual([item["text"] for item in segments if item["dialogue"]], ["“Hello!”", '"Hi."'])

    def test_large_narration_is_batched_without_losing_text(self):
        text = "a" * (MAX_ANALYSIS_CHARS + 1) + ' "Hello."'
        segments = split_dialogue(text)
        batches = list(analysis_batches(segments))
        self.assertEqual("".join(item["text"] for batch in batches for item in batch), text)
        self.assertTrue(all(sum(len(item["text"]) for item in batch) <= MAX_ANALYSIS_CHARS for batch in batches))

    def test_speaker_attribution_rejects_missing_or_duplicate_ids(self):
        segments = split_dialogue('Alice said "hello" and Bob said "bye".')
        for entries in ([], [{"id": 1, "speaker": "Alice", "delivery": ""}] * 2):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                identify_speakers(segments, analysis_client({"speakers": entries}), "model")

    def test_analyze_epub_and_validate_assignments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.epub"
            book = epub.EpubBook()
            book.set_identifier("test-book")
            book.set_title("Test Book")
            book.add_author("Test Author")
            chapter = epub.EpubHtml(title="Chapter One", file_name="chapter.xhtml")
            chapter.content = '<h1>Chapter One</h1><p>Alice said, "Hello."</p>'
            book.add_item(chapter)
            book.add_item(epub.EpubNcx())
            book.spine = [chapter]
            epub.write_epub(str(path), book)
            client = analysis_client({"speakers": [{"id": 1, "speaker": "Alice", "delivery": "cheerful"}]})
            plan = create_plan(path, 1, client, "test-model", "af_bella")
            self.assertEqual(plan["book_title"], "Test Book")
            self.assertEqual(plan["voices"]["Alice"], "")
            self.assertEqual(next(segment["text"] for segment in plan["segments"] if segment["dialogue"]), '"Hello."')
            with self.assertRaisesRegex(ValueError, "Alice"):
                validate_plan(plan)
            plan["voices"]["Alice"] = "af_sarah"
            validate_plan(plan)
            with self.assertRaisesRegex(ValueError, "Chapter must be"):
                create_plan(path, 2, client, "test-model", "af_bella")

    def test_render_uses_selected_voices_and_tags(self):
        plan = {
            "version": 1, "title": "Chapter", "book_title": "Book", "author": "Author",
            "chapter": 2, "voices": {"narrator": "af_bella", "Alice": "af_sarah"},
            "segments": [
                {"text": "Alice said.", "speaker": "narrator"},
                {"text": '"Hello."', "speaker": "Alice"},
                {"text": '"Who?"', "speaker": "unknown"},
            ],
        }
        source = io.BytesIO()
        Sine(440).to_audio_segment(duration=120).export(source, format="mp3")
        client = Mock()
        client.audio.speech.create.return_value = SimpleNamespace(content=source.getvalue())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "chapter.mp3"
            render_plan(plan, output, client)
            self.assertGreater(len(AudioSegment.from_mp3(output)), 0)
            tags = ID3(output)
            self.assertEqual(str(tags["TIT2"]), "Chapter")
            self.assertEqual(str(tags["TRCK"]), "2")
            self.assertEqual(
                [call.kwargs["voice"] for call in client.audio.speech.create.call_args_list],
                ["af_bella", "af_sarah", "af_bella"],
            )
            with self.assertRaisesRegex(ValueError, "already exists"):
                render_plan(plan, output, client)

    @unittest.skipUnless(os.environ.get("KOKORO_INTEGRATION") == "1", "requires a running Kokoro server")
    def test_real_kokoro_render(self):
        from openai import OpenAI

        plan = {
            "version": 1, "title": "Sample", "book_title": "Test", "author": "Test",
            "chapter": 1, "voices": {"narrator": "af_bella", "Alice": "af_sarah"},
            "segments": [
                {"text": "Alice stepped inside.", "speaker": "narrator"},
                {"text": '"Hello there."', "speaker": "Alice"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sample.mp3"
            render_plan(plan, output, OpenAI())
            self.assertGreater(len(AudioSegment.from_mp3(output)), 1000)
            self.assertEqual(str(ID3(output)["TIT2"]), "Sample")


if __name__ == "__main__":
    unittest.main()
