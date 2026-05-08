# Demo Guide

This guide gives a safe way to evaluate **Semantic Video Search** locally. The commands favor help screens, mock/dry-run paths, or clearly labeled local execution so the demo stays honest.

## Quick Orientation

Start with the CLI help and README sections before running any external-service path.

```bash
video-search --help
```
```bash
video-search index --help
```
```bash
video-search search --help
```

If a command needs live services or credentials, run the help command first and configure only the services you actually intend to test.

## Portfolio Walkthrough

Use this sequence in an interview or portfolio review:

1. Open the README and explain the problem the project solves in one sentence.
2. Open [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) and walk through the data flow.
3. Show the relevant model fields or tests that label mock, fallback, degraded, or warning states.
4. Run the local test suite or the project CI page to show the implementation is maintained.
5. Explain one tradeoff or limitation from the README instead of overselling the project.

## Suggested Demo Script

- **Problem**: Video indexing pipeline that extracts frames with ffmpeg, captions/embeds them with Gemini, and stores searchable frame vectors in LanceDB with explicit test-store boundaries.
- **Engineering signal**: the project models external-service failure instead of hiding it.
- **Safety signal**: generated or assisted outputs are explicitly marked for human review.
- **Portfolio signal**: the Git history includes focused maintenance PRs, CI fixes, and docs polish.

## Screenshots And Videos

No generated screenshot or video is included here because a fake recording would weaken the portfolio. If you add one later, capture it from a real local run with sanitized sample data and include the exact command/config used to produce it.

## Demo Boundaries

- Quality depends on frame sampling and caption quality.
- Large videos need storage/runtime planning.
- Gemini and LanceDB setup are required for full indexing behavior.
