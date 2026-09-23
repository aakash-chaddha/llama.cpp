#pragma once

// Parallel constrained decisions for finite JSON schemas, shared by llama-parallel-decision
// and llama-server's /decision endpoint.
//
// Every field of a schema has a finite set of allowed values. After a shared context, each
// field's value is scored as token paths following that field's own suffix; every scored path
// runs as its own sequence forked from the context (llama_memory_seq_cp), so all fields are
// evaluated in one batched llama_decode and cannot see each other. Small fields score every
// divergence node of their token trie at once and return the exact constrained distribution;
// larger fields walk the trie greedily.

#include "llama.h"
#include "json.h"

#include <functional>
#include <string>
#include <utility>
#include <vector>

struct common_chat_templates;

namespace llama_decision {

using tokens_t = std::vector<llama_token>;

// One field as the scorer sees it: the text before its value and the allowed value texts.
struct field_input {
    std::string              suffix;     // e.g.  '  "fire": '
    std::vector<std::string> candidates; // allowed values, with the suffix's shared prefix removed
};

struct options {
    std::string mode           = "auto"; // auto: tree up to tree_max values, else greedy; tree; greedy
    size_t      tree_max       = 128;
    bool        split_boundary = false;  // legacy: tokenise suffix and values separately
    bool        allow_cache    = true;   // reuse the cached static prefix when it matches
};

struct field_result {
    int                winner       = -1;
    float              path_score   = 1.0f;
    int                scored_nodes = 0;
    bool               tree         = false;
    std::vector<float> probs;            // tree fields: probability of every allowed value
};

struct result {
    std::vector<field_result> fields;
    bool   cache_hit      = false;
    size_t shared_tokens  = 0;
    size_t context_tokens = 0;
    int    rows           = 0;
    int    rounds         = 0;
    double prefill_ms     = 0;
    double scoring_ms     = 0;
};

// A context is a list of alternating segments: text the engine tokenises itself, or a reference to
// media the caller has already encoded elsewhere. Media is decoded through a media_decoder callback
// (the engine itself does not know about images, only about an integer index and a position).
struct context_segment {
    enum kind_t { TEXT, MEDIA };

    kind_t   kind  = TEXT;
    tokens_t toks;      // TEXT: tokens to decode causally at the running position
    int      media = -1; // MEDIA: index into the caller's list of encoded media

    static context_segment of_text(tokens_t t) {
        context_segment s;
        s.kind = TEXT;
        s.toks = std::move(t);
        return s;
    }
    static context_segment of_media(int index) {
        context_segment s;
        s.kind  = MEDIA;
        s.media = index;
        return s;
    }
};

// Decodes media `media_index` into (seq, pos) on the engine's own context and returns the position
// after it, exactly like mtmd_helper_decode_image_chunk's new_n_past (M-RoPE may advance by a
// different amount than the token count). Called from the engine's thread, one sequence at a time.
using media_decoder = std::function<llama_pos(llama_seq_id seq, llama_pos pos, int media_index)>;

// Several contexts decided against one schema and one cached prefix. Items carry fields,
// context_tokens and rows; timings and cache state cover the whole batch.
struct batch_result {
    std::vector<result> items;
    bool   cache_hit     = false;
    size_t shared_tokens = 0;
    int    rows          = 0;
    int    rounds        = 0;
    double prefill_ms    = 0;
    double scoring_ms    = 0;
};

// Scores decisions on an existing context with the sequence ids [seq_base, seq_base + n_seqs):
// one keeps the cached static prefix; the rest hold one trunk (prefix + context) per context in
// flight, then branches. The context needs a unified KV cache so branches share the trunk's cells.
class engine {
  public:
    engine(llama_context * ctx, llama_seq_id seq_base, int n_seqs);

    result decide(const std::string & shared_text, const std::string & context_text,
                  const std::vector<field_input> & fields, const options & opt);

    // Contexts are prefilled together and their branches scored together, in groups sized to fit
    // the sequence budget; results keep the order of the contexts.
    batch_result decide_batch(const std::string & shared_text, const std::vector<std::string> & contexts,
                              const std::vector<field_input> & fields, const options & opt);

    // Same, but every context is a list of text/media segments. Text is prefilled in packed batches,
    // one round at a time; each media segment is decoded through `decode_media` on its own sequence,
    // so the caller owns M-RoPE positions and the non-causal span. `decode_media` may be empty when
    // no context has a media segment.
    batch_result decide_batch_media(const std::string & shared_text,
                                    const std::vector<std::vector<context_segment>> & contexts,
                                    const std::vector<field_input> & fields, const options & opt,
                                    const media_decoder & decode_media);

  private:
    struct prompt_part {
        const tokens_t * toks;
        llama_pos        pos0;
        llama_seq_id     seq;
    };
    struct branch {
        llama_seq_id trunk;
        llama_pos    pos0;
        tokens_t     toks;
        tokens_t     cands;
    };

    llama_context     * ctx;
    const llama_vocab * vocab;
    llama_memory_t      mem;
    llama_seq_id        seq_snap, seq_pool;
    int                 n_pool;
    tokens_t            cached;

    void     decode_parts(const std::vector<prompt_part> & parts);
    bool     prepare_prefix(const tokens_t & shared, bool allow_cache);
    std::vector<std::vector<float>> score_branches(const std::vector<branch> & branches, llama_seq_id first, int n_free);
};

// ---- schema compiler (the C++ counterpart of llama-mojo's tools/prepare_decisions.py)

struct field_spec {
    std::string              name;
    std::string              type;        // boolean | enum | integer | number
    std::string              description;
    std::string              aggregate;   // mode | median | mean (median/mean: numeric fields)
    std::vector<common_json> values;      // typed values; index = candidate index
    std::vector<double>      numbers;     // numeric fields: the same values as doubles
    std::vector<std::string> encoded;     // JSON text of each value
};

struct compiled_schema {
    std::string              system_text; // fixed instructions + field catalogue (cacheable)
    std::vector<field_spec>  specs;
    std::vector<field_input> inputs;
};

// Accepts compact field specs {"name": {"type": ..., "description": ..., ...}} or a JSON Schema
// object with "properties" (boolean, string+enum, integer min/max, number min/max/multipleOf).
compiled_schema compile_schema(const common_json & schema, const std::string & instructions);

// Tokenises a context string the way the engine does: parse_special, a single BOS (a chat template
// may already start with the BOS text). `add_special` is false when a cached prefix precedes it.
// Shared with the server so the text-only path and the text segments of a media context cannot drift.
tokens_t tokenize_text(const llama_vocab * vocab, const std::string & text, bool add_special);

// Renders system + user messages with the model's chat template (thinking disabled) and splits
// the prompt into the static prefix (cached across requests) and the per-request part: the
// context, the generation prompt and the opening brace of the JSON answer. `media_markers` are
// appended to the user message after the sentinel, so they land at the end of the per-request part
// (text first, then media - the order the caller sent it).
std::pair<std::string, std::string> render_prompt(const common_chat_templates * tmpls, bool use_jinja,
                                                  const std::string & system_text, const std::string & context,
                                                  const std::vector<std::string> & media_markers = {});

// {"decision": {...}, "fields": {...}} from the scores, applying each numeric field's aggregate.
common_json assemble(const compiled_schema & cs, const result & r);

} // namespace llama_decision
