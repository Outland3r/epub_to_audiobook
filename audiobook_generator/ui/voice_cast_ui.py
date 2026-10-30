"""Reviewable character voices in the WebUI."""

import copy
import json
import os
from pathlib import Path

import gradio as gr
from openai import OpenAI, OpenAIError

from audiobook_generator.core.voice_cast import create_plan, render_plan, validate_plan


def plan_tables(plan, available_voices=()):
    if not isinstance(plan, dict) or not isinstance(plan.get("voices"), dict) or not isinstance(plan.get("segments"), list):
        raise ValueError("Invalid voice plan: expected voices and segments")
    checked = copy.deepcopy(plan)
    if not all(isinstance(name, str) and isinstance(voice, str) for name, voice in checked["voices"].items()):
        raise ValueError("Invalid voice plan: voice names and values must be strings")
    checked["voices"]["narrator"] = checked["voices"].get("narrator") or "unassigned"
    for segment in checked["segments"]:
        if isinstance(segment, dict) and isinstance(segment.get("speaker"), str):
            speaker = segment["speaker"]
            if speaker.casefold() != "unknown":
                checked["voices"][speaker] = checked["voices"].get(speaker) or "unassigned"
    validate_plan(checked)

    voices = copy.deepcopy(plan["voices"])
    voices.setdefault("narrator", "")
    narrator = voices["narrator"]
    choices = [voice for voice in available_voices if voice != narrator]
    for speaker in dict.fromkeys(segment["speaker"] for segment in plan["segments"]):
        if speaker.casefold() == "unknown":
            continue
        if speaker not in voices:
            voices[speaker] = ""
        if not voices[speaker] and speaker != "narrator" and choices:
            voices[speaker] = choices.pop(0)
    segments = [
        [str(index), segment["text"], segment["speaker"], segment.get("delivery", "")]
        for index, segment in enumerate(plan["segments"])
    ]
    return [[name, voice] for name, voice in voices.items()], segments


def reviewed_plan(plan, voice_rows, segment_rows):
    if plan is None:
        raise ValueError("Analyze a chapter or load a saved plan first")
    plan_tables(plan)
    if not isinstance(voice_rows, list) or not isinstance(segment_rows, list):
        raise ValueError("Voice and segment tables must contain rows")
    voices = {}
    for row in voice_rows:
        if len(row) != 2 or not isinstance(row[0], str) or not row[0].strip():
            raise ValueError("Each voice row needs a speaker name")
        name = row[0].strip()
        if name in voices:
            raise ValueError(f"Duplicate speaker in voices: {name}")
        if row[1] is not None and not isinstance(row[1], str):
            raise ValueError(f"Voice for {name} must be a string")
        voices[name] = (row[1] or "").strip()
    updated = copy.deepcopy(plan)
    updated["voices"] = voices
    seen = set()
    for row in segment_rows:
        if len(row) != 4 or not isinstance(row[0], str) or not row[0].isdigit():
            raise ValueError("Each segment needs its original ID, text, speaker, and delivery")
        index = int(row[0])
        if index in seen or index >= len(updated["segments"]):
            raise ValueError(f"Invalid or duplicate segment ID: {row[0]}")
        segment = updated["segments"][index]
        if row[1] != segment["text"]:
            raise ValueError(f"Segment {index} text was changed; only speaker and delivery may be edited")
        if not isinstance(row[2], str) or not row[2].strip():
            raise ValueError(f"Segment {index} needs a speaker")
        if row[3] is not None and not isinstance(row[3], str):
            raise ValueError(f"Segment {index} delivery must be text")
        segment["speaker"] = row[2].strip()
        segment["delivery"] = (row[3] or "").strip()
        seen.add(index)
    if len(seen) != len(updated["segments"]):
        raise ValueError("All original segments must remain in the review table")
    return updated


def output_path(directory, filename, suffix):
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("Set an output directory")
    if not isinstance(filename, str) or not filename or Path(filename).name != filename or not filename.lower().endswith(suffix):
        raise ValueError(f"Enter a filename ending in {suffix}, without directories")
    return Path(directory) / filename


def analyze_chapter(input_file, chapter, model, narrator_voice, available_voices):
    if not input_file:
        raise gr.Error("Select an EPUB at the top of the page")
    if not isinstance(chapter, (int, float)) or chapter < 1 or int(chapter) != chapter:
        raise gr.Error("Chapter must be a positive whole number")
    if not model or not model.strip():
        raise gr.Error("Enter an OpenRouter model or set OPENROUTER_MODEL")
    if not narrator_voice:
        raise gr.Error("Select a narrator voice")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise gr.Error("Set OPENROUTER_API_KEY and restart the WebUI")
    try:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        plan = create_plan(input_file, int(chapter), client, model.strip(), narrator_voice)
        voice_rows, segment_rows = plan_tables(plan, available_voices)
    except (ValueError, OSError, OpenAIError) as error:
        raise gr.Error(f"Analysis failed: {error}") from error
    return plan, voice_rows, segment_rows, f"Chapter {plan['chapter']}: {plan['title']}. Review the speaker and voice assignments."


def load_plan(file_path):
    if not file_path:
        raise gr.Error("Select a JSON voice plan to load")
    try:
        with open(file_path, encoding="utf-8") as file:
            plan = json.load(file)
        voice_rows, segment_rows = plan_tables(plan)
    except (ValueError, OSError) as error:
        raise gr.Error(f"Could not load plan: {error}") from error
    return plan, voice_rows, segment_rows, f"Loaded chapter {plan['chapter']}: {plan['title']}."


def save_plan(plan, voice_rows, segment_rows, directory, filename):
    try:
        updated = reviewed_plan(plan, voice_rows, segment_rows)
        path = output_path(directory, filename, ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as file:
            json.dump(updated, file, ensure_ascii=False, indent=2)
            file.write("\n")
    except (ValueError, OSError) as error:
        raise gr.Error(f"Could not save plan: {error}") from error
    return str(path), f"Plan saved to {path}. It contains the chapter text."


def render_chapter(plan, voice_rows, segment_rows, directory, filename):
    if not os.environ.get("OPENAI_BASE_URL"):
        raise gr.Error("Set OPENAI_BASE_URL to the Kokoro endpoint and restart the WebUI")
    try:
        updated = reviewed_plan(plan, voice_rows, segment_rows)
        validate_plan(updated)
        path = output_path(directory, filename, ".mp3")
        render_plan(updated, path, OpenAI())
    except (ValueError, OSError, OpenAIError) as error:
        raise gr.Error(f"Rendering failed: {error}") from error
    return str(path), str(path), f"Rendered chapter to {path}."


def build_voice_cast_ui(input_file, output_dir, available_voices):
    with gr.Accordion("Experimental: character voices", open=False):
        gr.Markdown(
            "Analyze one chapter with OpenRouter, review speakers and Kokoro voices, then render. "
            "**Analysis sends chapter text to OpenRouter and its model provider and may incur charges.** "
            "Delivery notes are informational; Kokoro does not apply them. "
            "The normal Start button above does not use this workflow."
        )
        plan_state = gr.State()
        with gr.Row():
            chapter = gr.Number(label="Chapter number", value=1, precision=0, minimum=1)
            model = gr.Textbox(label="OpenRouter model", value=os.environ.get("OPENROUTER_MODEL", ""), placeholder="provider/model")
            narrator = gr.Dropdown(available_voices, value="af_bella" if "af_bella" in available_voices else None, label="Narrator voice", allow_custom_value=True)
        status = gr.Textbox(label="Character voice status", interactive=False)
        voice_table = gr.Dataframe(headers=["Speaker", "Kokoro voice"], datatype="str", type="array", row_count=(1, "dynamic"), col_count=(2, "fixed"), label="Voice assignments", interactive=True)
        segment_table = gr.Dataframe(headers=["ID", "Text (read-only)", "Speaker", "Delivery note"], datatype="str", type="array", row_count=(1, "dynamic"), col_count=(4, "fixed"), static_columns=[0, 1], label="Review segments", interactive=True, wrap=True)
        gr.Markdown("Use `unknown` for the narrator voice. To add a character, add a voice row and use that name in a segment.")
        with gr.Accordion("Available Kokoro voice IDs", open=False):
            gr.Markdown(", ".join(f"`{voice}`" for voice in available_voices))
        with gr.Row():
            gr.Button("Analyze chapter", variant="primary").click(
                fn=lambda book, number, llm, voice: analyze_chapter(book, number, llm, voice, available_voices),
                inputs=[input_file, chapter, model, narrator],
                outputs=[plan_state, voice_table, segment_table, status],
            )
            plan_upload = gr.File(label="Load saved voice plan", file_types=[".json"], type="filepath")
            gr.Button("Load plan").click(load_plan, inputs=[plan_upload], outputs=[plan_state, voice_table, segment_table, status])
        with gr.Row():
            plan_name = gr.Textbox(label="Plan filename", value="voice-plan.json")
            plan_download = gr.File(label="Saved plan", interactive=False)
            gr.Button("Save reviewed plan").click(save_plan, inputs=[plan_state, voice_table, segment_table, output_dir, plan_name], outputs=[plan_download, status])
        with gr.Row():
            audio_name = gr.Textbox(label="Audio filename", value="character-voices.mp3")
            audio_download = gr.File(label="Rendered MP3", interactive=False)
            audio_player = gr.Audio(label="Chapter preview", interactive=False)
            gr.Button("Render chapter", variant="primary").click(render_chapter, inputs=[plan_state, voice_table, segment_table, output_dir, audio_name], outputs=[audio_download, audio_player, status])
