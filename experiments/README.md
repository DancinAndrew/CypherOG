# Experiments

Retrieval setups for CypherOG compared on five fixed questions (`QUESTIONS` in `experiment_runner.py`). Every number below comes from files in this folder; `metrics.md` gives the full per-run table and metric definitions.

## What is here

| Path | What it is |
|---|---|
| `combo_a/` … `combo_e/`, `summary.md` | **Round 1** (April 2026): the original outputs for setups A–E, kept unchanged as the record. `experiment_runner.py` refuses to write here. |
| `corpus_round1.json` | The 39 documents the round-1 index contained, with the evidence (below). |
| `corpus_pin.json` | SHA-256 of every committed `data/processed` text; experiments refuse to run if the text changed. |
| `round1_reproduction_run/`, `round1_reproduction.md` | Round-1 retrieval re-run with today's code (no LLM) and compared cell by cell with the recorded files: **25/25 match**. |
| `gold.json` | The **pre-registered** metric definitions (member aliases for Q2, Juste Debout fact groups for Q4), committed before any round-2 run. |
| `round2/<corpus>_corpus/run<N>/` | **Round 2** (November 2026): E′ (control), F (hybrid), G (query expansion), three runs on each corpus. `round2/environment.txt` lists the package versions. |
| `metrics.md`, `metrics.json` | Output of `python rag_eval.py`: every run scored with `gold.json`, plus a stability table across repeats. |

## Setups

| Setup | Chunking | Retrieval | Prompt |
|---|---|---|---|
| A | naive, 500 characters, no overlap | dense top-5, no distance cutoff | weak |
| B | paragraph-aware, 600 / overlap 100 | dense top-5, cutoff 0.65 | weak |
| C | paragraph-aware, 1200 / overlap 200 | dense top-3, cutoff 0.65 | strict, XML tags |
| D | paragraph-aware, 300 / overlap 50 | dense top-10 cut to 5 (identical to top-5; the round-1 files call this "rerank", but no reranker existed), cutoff 0.5 | strict, XML tags |
| E / E′ | paragraph-aware, 600 / overlap 100 | dense top-5, cutoff 0.65 | chain-of-thought, XML tags |
| F | as E | BM25 + dense, reciprocal rank fusion (k = 60, equal weights); cutoff on the dense channel only | chain-of-thought |
| G | as E | LLM rewrites the question into up to 3 sub-queries; results merged keeping each chunk's smallest distance | chain-of-thought |

D differs from A in five settings at once (chunker, chunk size, top-k, cutoff, prompt), so its distance gap cannot be attributed to any one of them.

## How round 1 was pinned down

The repository holds 42 documents, but round 1 indexed 39. Rebuilding round-1 retrieval without `popping_glossary.md`, `popping_history_timeline.md` and `popping_style_taxonomy.md` reproduces all 25 recorded retrievals (same chunks, same order, distances within 0.0005). With all 42 documents only 11 of 25 match, and every mismatch is exactly those three documents' chunks being inserted with nothing else changed, which rules out approximate-search noise. `rag_eval.py --check-reproduction` performs this comparison.

## Metrics (pre-registered)

Mean cosine distance cannot evaluate F or G: for a fixed k, the dense top-k is already the set with the smallest total distance, so fusion or expansion can only keep or raise it. Two content metrics were therefore fixed in `gold.json` before round 2, on the questions where round 1 differed or failed:

- **Q2** (Electric Boogaloos members): how many of the nine notable members listed in the corpus the answer names.
- **Q4** (Juste Debout rules and history): how many of the top-k chunks mention Juste Debout together with a corpus fact (format, city, year, venue, organizer), and how many fact groups the answer states.

## Results

| Setup | Corpus | Runs | Mean cosine distance | Q2 members named (/9) | Q4 Juste Debout chunks | Q4 fact groups |
|---|---|---|---|---|---|---|
| A (round 1) | 39 docs | 1 | 0.399 | 9 | 0/5 | 0 |
| B (round 1) | 39 docs | 1 | 0.386 | 9 | 0/5 | 0 |
| C (round 1) | 39 docs | 1 | 0.384 | 3 | 0/3 | 0 |
| D (round 1) | 39 docs | 1 | 0.359 | 2 | 0/5 | 0 |
| E (round 1) | 39 docs | 1 | 0.386 | 9 | 0/5 | 0 |
| E′ | 39 docs | 3 | 0.386 | 9 | 0/5 | 0 |
| F | 39 docs | 3 | 0.443 | 3 | 2/5 | 2–3 |
| G | 39 docs | 3 | 0.466–0.474 | 0–2 | 1/5 | 1–2 |
| E′ | 42 docs | 3 | 0.379 | 9 | 1/5 | 3 |
| F | 42 docs | 3 | 0.460 | 4 | 1/5 | 3 |
| G | 42 docs | 3 | 0.463–0.471 | 1–2 | 1/5 | 1–2 |

Round 1 answers came from gemini-2.5-flash through a course proxy; round 2 used gpt-4.1-mini-2025-04-14. Ranges are min–max across the three runs.

**What the numbers show**

1. **Lower distance did not mean better answers.** D had the lowest mean distance (10% below A) but named 2 members to A's 9: its 300-character chunks split the member list, and the fragment holding it did not reach the top 5.
2. **Hybrid retrieval recovers exact-name content that dense retrieval misses.** On the same 39 documents, F was the only setup whose answers stated Juste Debout facts, in all three runs; none of them covered the event's rules. The decisive chunk (`Juste Debout.txt` #0) sits at cosine distance 0.798 from the question, outside the reach of dense retrieval and its 0.65 cutoff; BM25 found it by matching the name.
3. **Fusion rewards agreement, which cost list completeness.** The member-list chunk ranked 5th in dense retrieval but outside BM25's top 15, so its fused score (1/65) lost to chunks both retrievers found; the Suga Pop chunk, 15th in dense retrieval but 6th in BM25, took a top-5 slot. F named 3 members instead of 9.
4. **Query expansion was weaker and less stable.** G recovered one Juste Debout chunk but named 0–2 members, varying across runs.
5. **Curated data beat retrieval tricks here.** With the three Chinese notes added (42 documents), plain dense retrieval (E′) found Juste Debout facts too, while keeping all 9 members.

The default therefore stays E (dense, chain-of-thought); hybrid retrieval and query expansion are opt-in flags.

## Caveats

- Five questions, of which only two have pre-registered scores: these results illustrate mechanisms and trade-offs, not general performance.
- The embedding model reads only the first 128 tokens of each chunk (about 78% of a 600-character chunk on average), while BM25 indexes the whole chunk. Part of F's gain may come from BM25 seeing text the embedding never read.
- BM25 tokenizes Chinese character by character, so common characters match almost every Chinese chunk.
- The gold alias "Pete" (for Popin' Pete) is loose. No round-2 answer relies on it; in round 1 only D's answer does ("Pete" listed as a group member).
- All Q4 fact groups counted in the 18 round-2 answers also appear in the retrieved text, so none came from the model's own knowledge.
- Round-1 E and round-2 E′ differ in the answering model, so compare F and G with E′, not with round 1.

## Reproduce

```bash
# Round-1 retrieval, no LLM; then compare with the recorded files
python experiment_runner.py --combos a b c d e --corpus round1 --retrieval-only --out experiments/round1_reproduction_run
python rag_eval.py --check-reproduction experiments/round1_reproduction_run

# Round 2 (needs an LLM key in .env)
LITELLM_MODEL=gpt-4.1-mini-2025-04-14 python experiment_runner.py --combos e f g --corpus round1 --out experiments/round2/round1_corpus/run1

# Score everything with gold.json
python rag_eval.py
```
