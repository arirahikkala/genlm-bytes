# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GenLM Bytes is a Python library for byte-level language modeling. It converts token-level language models into byte-level language models using a beam search algorithm over a trie data structure. Requires Python >= 3.11.

## Commands

```bash
# Install for development
pip install -e ".[test,docs]"

# Run all tests
pytest tests

# Run a single test
pytest tests/test_beam.py::test_basics

# Run tests with coverage
pytest tests --cov=genlm/bytes --cov-report=json

# Lint/format (via pre-commit hooks)
pre-commit run --all-files

# Build docs
mkdocs build

# Serve docs locally
mkdocs serve
```

## Architecture

The library lives in `genlm/bytes/` and has two main layers:

### Trie layer (`trie.py`)
- **`TokenByteTrie`** — Builds a trie from a token vocabulary where each token is decomposed into its byte sequence. Nodes represent byte prefixes; leaves represent complete tokens. Uses sparse matrix multiplication (`M_no_eos`, `M_with_eos`) to efficiently propagate token-level weights to every trie node via `weight_sum()` / `weight_max()`.
- **`AsyncTokenByteTrie`** — Async wrapper that automatically batches concurrent `weight_sum`/`weight_max` calls via a background queue, grouping by `(TrieOp, TrieMode)` pairs.
- **`TrieMode`** — Enum controlling EOS handling: `WITHOUT_EOS` treats EOS tokens as normal (used during prefill); `WITH_EOS` aggregates EOS token probability to a special EOS node (used during generation).

### Beam search layer (`byte_lm/`)
- **`ByteBeamState`** (`beam.py`) — Main entry point. Maintains a beam of `LazyTrieState` candidates. The `<<` operator advances all candidates by one byte, `prune()` removes low-probability candidates, `logp_next()` returns the marginal byte distribution, and `prefill()` conditions on a byte sequence. `extend()` commits partial tokens (EOT transitions) to start new tokens.
- **`LazyTrieState`** (`trie_state.py`) — Single candidate in the beam. Tracks position in the trie (`node`), cumulative log-probability (`weight`), and lazily-computed node masses. `materialize()` calls the LM and trie to compute masses. The `<<` operator navigates trie edges; `extend()` commits the current partial token.
- **`StatefulTokenizedLM`** (`lm_state.py`) — Wraps `genlm.backend.AsyncLM` with token-level context tracking. The `<<` operator appends a token ID; `logp_next()` gets next-token log-probs.
- **`TokenHealer`** (`heal.py`) — Adaptive token healing: when no trie edge exists for a byte, tries backing off (committing a shorter prefix as a token) and replaying the remaining bytes from root.

### Key constants and conventions
- Byte values 0–255 are regular bytes, 256 is EOT (end-of-token, internal trie marker), 257 is EOS (end-of-sequence), 258+ are special token virtual bytes.
- The `<<` operator is used throughout for "advance by one unit" (byte for beam/trie states, token for LM state).
- Most operations are async; tests use `pytest-asyncio`.
- The `genlm-backend` package (separate repo) provides `AsyncLM` and tokenizer utilities.

## Code Style

- Formatting and linting handled by **ruff** (via pre-commit hooks).
- No additional style configuration beyond what ruff enforces.
