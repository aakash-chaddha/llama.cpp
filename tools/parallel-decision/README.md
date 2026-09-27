# parallel-decision

Answer a finite JSON schema in one batched forward pass, instead of generating the JSON token by token.

Every field of a schema has a fixed set of allowed values (enums, booleans, bounded integers, numbers on a grid).
After the context, each field's allowed values are scored as token paths that fork from the same KV cache, so all
fields are answered in one `llama_decode` and cannot see each other. Each answer comes back with a probability, and
the JSON object is assembled by code, so it always matches the schema.

This directory holds the engine (`decision-engine.*`), a CLI (`llama-parallel-decision`), and the engine is also
served by `llama-server` as `POST /v1/decision`.

## Build

Same as llama.cpp:

```bash
cmake -B build -DGGML_CUDA=ON        # or plain `cmake -B build` for CPU / Metal
cmake --build build --config Release -j
```

## Run the server

`--decision-seqs N` reserves the sequence slots the decisions need: one holds the cached instructions, one per
context in flight, the rest are the parallel questions (at least 3). It also switches the KV cache to unified, which
is what lets the branches share the context's cells. `--ubatch-size` has to be at least as large as the largest image,
because an image is decoded in one non-causal batch, and media contexts need `--mmproj` - without it they are a 400.

```bash
./build/bin/llama-server -m models/gemma-4-12b-it-UD-Q4_K_XL.gguf \
  --mmproj models/gemma-4-12b-mmproj-F16.gguf -ngl 99 -fa on -c 32768 --parallel 1 \
  --ubatch-size 2048 --cache-type-k q8_0 --cache-type-v q8_0 --jinja \
  --decision-seqs 12 --port 8096
```

| flag | why |
|---|---|
| `--decision-seqs N` | enables `/v1/decision` and reserves its sequences; also enables the unified KV cache |
| `--ubatch-size N` | must be >= the token count of the largest image, or the request fails with `400 image needs N tokens, raise --ubatch-size to at least N` |
| `--mmproj FILE` | the vision/audio tower; media contexts need it |
| `--cache-type-k/v q8_0` | a 32k context on a 24 GB card: 12.5 GB against 15.5 GB with f16 |
| `--jinja` | the chat template the decision prompt is rendered with |
| `-fa on` | required for a quantized V cache, and the fastest prefill |
| `--parallel 1` | decisions are sized from `--decision-seqs`, not from the slot count |

With a presets file, one process serves chat and decisions and can hold more than one model:

```ini
[*]
ngl = 99
fa = on
jinja = 1
parallel = 1
cache-type-k = q8_0
cache-type-v = q8_0
decision-seqs = 24
ubatch-size = 2048

[gemma-4-12b]
model = ./models/gemma-4-12b-it-UD-Q4_K_XL.gguf
mmproj = ./models/gemma-4-12b-mmproj-F16.gguf
ctx-size = 32768
decision-seqs = 12

[qwen3.5-0.8b]
model = ./models/Qwen3.5-0.8B-UD-IQ2_M.gguf
ctx-size = 32768
```

```bash
./build/bin/llama-server --models-preset models.ini --models-max 2 --port 8096
```

`--models-max 2` keeps two models resident, so a client can switch between them without a reload: the router lists
every preset in `GET /v1/models` and each request picks one with its `model` field.

The encoder cache of media contexts is on by default; `--no-decision-media-cache` (or `LLAMA_ARG_DECISION_MEDIA_CACHE=0`)
turns it off, and a request can opt out with `"cache_media": false`.

How many sequences a model affords depends on its attention. A plain-attention model shares the context's cells, so
128 sequences cost almost nothing. A sliding-window model (Gemma) allocates its window per sequence, so keep it low
(12 on a 12 GB card). Hybrid models with recurrent layers work, but llama.cpp splits their batches per sequence
length, so branches run in several passes instead of one.

## POST /v1/decision

`contexts` is a list of 1-256 contexts. They share one schema, one set of instructions, and one cached prefix; results
come back in the same order. A context is either a plain string, or an array of content parts (see
[Media contexts](#media-contexts)).

```bash
curl http://localhost:8096/v1/decision -H "Content-Type: application/json" -d '{
  "model": "gemma-4-12b",
  "instructions": "Answer each question about this support request from its state.",
  "schema": {
    "category": {"type": "enum", "choices": ["billing","technical","cancellation","other"],
                 "description": "What type of support request is this?"},
    "urgent":   {"type": "boolean", "description": "Does this need urgent handling?"},
    "priority": {"type": "enum", "choices": ["low","medium","high","critical"],
                 "description": "Rate support priority."}
  },
  "contexts": ["I was charged twice and need this fixed today."]
}'
```

```json
{
  "object": "decision",
  "results": [
    {
      "decision": {"category": "billing", "urgent": true, "priority": "high"},
      "fields": {
        "category": {"value": "billing",  "probability": 1.0,  "scored_nodes": 1, "tree": true},
        "urgent":   {"value": true,       "probability": 1.0,  "scored_nodes": 1, "tree": true},
        "priority": {"value": "high",     "probability": 0.74, "scored_nodes": 1, "tree": true}
      },
      "usage": {"context_tokens": 21, "scored_rows": 14}
    }
  ],
  "usage": {"prompt_tokens": 137, "cached_tokens": 116, "context_tokens": 21, "scored_rows": 14},
  "timings": {"prefill_ms": 50.7, "scoring_ms": 50.0, "total_ms": 100.7, "rounds": 1, "per_decision_ms": 100.7}
}
```

(That response is a real one: Gemma 4 12B on an RTX 3060, warm cache.)

### Schema

Compact fields, or a JSON Schema object with `properties`:

| type | keys | notes |
|---|---|---|
| `enum` | `choices` (or `enum`) | 1-255 values |
| `boolean` | - | true / false |
| `integer` | `minimum`, `maximum` | 1-255 values |
| `number` | `minimum`, `maximum`, `step` (`multipleOf` in JSON Schema) | fixed-width decimals |

Numeric fields take `aggregate`: `mode` (default), `median` or `mean`.

### Options

| field | default | meaning |
|---|---|---|
| `instructions` | `""` | prepended to the generated field catalogue; cached with it |
| `mode` | `auto` | `tree` scores every divergence node and returns exact probabilities; `greedy` walks the trie; `auto` picks tree up to `tree_max` values |
| `tree_max` | 128 | per-field switch between tree and greedy |
| `cache_prompt` | true | reuse the cached instructions + schema prefix |
| `cache_media` | server default | reuse the encoder output of a media file this server has already seen (`--decision-media-cache` / `--no-decision-media-cache` sets the default) |

### Media contexts

A context can be a string, or an array of content parts: any number of `text`, `image_url` and `input_audio`
parts. An image must be a base64 `data:image/...` URL; audio is raw base64 or a `data:` URL (mtmd auto-detects
the format from the magic bytes). Video is rejected with a 400 - mtmd has no video chunk type, so send frames as
`image_url` parts. The model needs an `mmproj` (`--mmproj`) carrying the matching encoder. This is the OpenAI
content-part shape the chat endpoint already takes, so a multimodal client can address `/v1/decision` directly:

```json
{
  "model": "gemma-4-12b",
  "instructions": "Decide from the screenshots and the call recording. The media is the evidence.",
  "schema": {
    "type":  {"type": "enum", "choices": ["animal", "person", "object"], "description": "The main subject."},
    "legs":  {"type": "integer", "minimum": 0, "maximum": 5, "description": "How many legs are visible."}
  },
  "contexts": [[
    {"type": "text", "text": "The checkout screen, then the error, then the call."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}},
    {"type": "input_audio", "input_audio": {"data": "<base64 wav>", "format": "wav"}}
  ]]
}
```

The media is the evidence: every part is encoded once per request and decoded into the decision's trunk in part
order - markers are appended after the context text, so a context reads text first, then its media - before the
branches fork. Every field is scored conditioned on the pixels or the waveform itself, no caption is generated in
between, and the whole request is one pass. `usage.media_tokens` (per result and per request) reports how many
memory cells the media occupied; `prompt_tokens` includes them.

A media part whose attention span is non-causal (an image, for every projector but Gemma's causal ones) has to fit
in one `ubatch`. If it does not, the request fails with `image needs N tokens, raise --ubatch-size to at least N`
instead of aborting the server - so set `--ubatch-size` (or `ubatch-size` in a preset) to at least the largest image
you expect. Causal media (audio) is split into batches by the mtmd helper and has no such limit. Text parts keep the
cached shared prefix, and a request can mix media contexts and plain-text contexts in any order.

The encoder output of a media file is cached between requests when `--decision-media-cache` is on (the default).
mtmd gives every file a bitmap id that is a SHA-256 of its bytes, and the same bytes always encode to the same
vectors whatever prompt surrounds them, so the id is the whole key. It is exact, but it is also small: on
gemma-4-12b it saves 5 ms on a 1024px image and 19 ms on a 2310x3072 one, because the encoder is not where the
time is - a repeat still pays the bitmap decode + preprocessing (~200 ms for that large image) and the media
decode into the trunk (210-345 ms). `usage.media_cached_tokens` reports the tokens served from the cache, and
`cache_media: false` in the request turns it off for that request.

## CLI

`llama-parallel-decision` runs the same engine from a worker process (stdin/stdout protocol, one JSON request per
line). Environment: `DECIDE_TREE`, `DECIDE_TREE_MAX`, `DECIDE_NSEQ`, `DECIDE_SPLIT_BOUNDARY`.

## A UI for it

[decision-playground](https://github.com/thecodacus/decision-playground) is a browser-only playground: it talks
straight to your llama-server, runs a decision and the same question as a chat completion side by side with live
timers, and has a small game whose agents decide through the endpoint.
