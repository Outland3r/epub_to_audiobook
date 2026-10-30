# Character voice roadmap

## Prototype

- [x] Analyze one EPUB chapter with OpenRouter using the existing chapter parser.
- [x] Save a reviewable plan with dialogue, suggested speakers, and delivery notes.
- [x] Assign Kokoro voices in the plan and render one tagged MP3 chapter.
- [x] Preserve the existing single-voice conversion workflow.
- [x] Cover segmentation, attribution validation, and rendering with automated tests.
- [x] Test rendering with a running Kokoro server and synthetic text.
- [ ] Test attribution with a real OpenRouter model.

## Future work

- [ ] Add AWS Bedrock as an alternative analysis provider.
- [ ] Add a WebUI for reviewing dialogue, editing assignments, and previewing voices.
- [ ] Track characters and aliases consistently across chapters and books.
- [ ] Preserve paragraph and scene boundaries throughout parsing and synthesis.
- [ ] Improve dialogue detection for nested quotations, unquoted dialogue, and languages
      using other quotation conventions.
- [ ] Evaluate speaker attribution against manually labeled chapters and handle
      ambiguity with confidence scores and review queues.
- [ ] Explore expressive delivery controls supported by the selected TTS provider;
      Kokoro delivery notes are not currently applied.
- [ ] Add resumable rendering, per-segment regeneration, and cost estimation.
- [ ] Evaluate pacing, pauses, transitions, pronunciation, and Audiobookshelf playback
      on representative books.
