"""Analyze and render a reviewable single-chapter voice plan."""

import argparse
import json
import os
from pathlib import Path

from audiobook_generator.core.voice_cast import create_plan, render_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="Analyze a chapter with OpenRouter")
    analyze.add_argument("epub", help="Input EPUB")
    analyze.add_argument("plan", help="Output JSON plan")
    analyze.add_argument("--chapter", type=int, required=True, help="1-based chapter index")
    analyze.add_argument("--model", required=True, help="OpenRouter model supporting JSON responses")
    analyze.add_argument("--narrator-voice", default="af_bella", help="Kokoro narrator voice")
    render = commands.add_parser("render", help="Render a reviewed plan using Kokoro")
    render.add_argument("plan", help="Reviewed JSON plan")
    render.add_argument("output", help="Output MP3")
    args = parser.parse_args()

    from openai import OpenAI

    if args.command == "analyze":
        path = Path(args.plan)
        if path.exists():
            parser.error("Plan already exists; choose a different output path")
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            parser.error("OPENROUTER_API_KEY is required for analysis")
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        plan = create_plan(args.epub, args.chapter, client, args.model, args.narrator_voice)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as file:
            json.dump(plan, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print(f"Plan saved to {path}. Review speakers and assign character voices before rendering.")
    else:
        if not os.environ.get("OPENAI_BASE_URL"):
            parser.error("Set OPENAI_BASE_URL to the Kokoro OpenAI-compatible endpoint")
        with open(args.plan, encoding="utf-8") as file:
            plan = json.load(file)
        render_plan(plan, args.output, OpenAI())
        print(f"Rendered chapter to {args.output}")


if __name__ == "__main__":
    main()
