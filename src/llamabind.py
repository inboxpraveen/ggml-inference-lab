"""
Minimal cffi (ABI mode, no compiler needed) binding for llama.dll from the prebuilt llama.cpp b10941 release.

The point of this file is that it is small: a few structs copied from include/llama.h and ~45 functions.
Everything the driver does (batched decode, zero-copy logits, argmax sampling, self-speculative n-gram
drafting, KV-cache rollback) is built on these calls. If llama.h changes, diff it against this file.
"""
import os
from cffi import FFI

ffi = FFI()
ffi.cdef(r"""
typedef int32_t llama_pos;
typedef int32_t llama_token;
typedef int32_t llama_seq_id;
struct llama_vocab; struct llama_model; struct llama_context; struct llama_sampler;
typedef struct llama_memory_i * llama_memory_t;
typedef struct ggml_backend_device * ggml_backend_dev_t;
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
typedef bool (*llama_progress_callback)(float progress, void * user_data);
typedef bool (*ggml_backend_sched_eval_callback)(void * t, bool ask, void * user_data);
typedef bool (*ggml_abort_callback)(void * data);
typedef void (*ggml_log_callback)(int level, const char * text, void * user_data);

enum llama_split_mode { LLAMA_SPLIT_MODE_NONE = 0, LLAMA_SPLIT_MODE_LAYER = 1, LLAMA_SPLIT_MODE_ROW = 2, LLAMA_SPLIT_MODE_TENSOR = 3 };
enum llama_load_mode  { LLAMA_LOAD_MODE_AUTO = -1, LLAMA_LOAD_MODE_NONE = 0, LLAMA_LOAD_MODE_MMAP = 1, LLAMA_LOAD_MODE_MLOCK = 2, LLAMA_LOAD_MODE_MMAP_MLOCK = 3, LLAMA_LOAD_MODE_DIRECT_IO = 4 };
enum llama_lazy_mode  { LLAMA_LAZY_MODE_OFF = 0, LLAMA_LAZY_MODE_AUTO = 1, LLAMA_LAZY_MODE_ON = 2 };
enum llama_context_type { LLAMA_CONTEXT_TYPE_DEFAULT = 0, LLAMA_CONTEXT_TYPE_MTP = 1 };
enum llama_rope_scaling_type { LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1, LLAMA_ROPE_SCALING_TYPE_NONE = 0, LLAMA_ROPE_SCALING_TYPE_LINEAR = 1, LLAMA_ROPE_SCALING_TYPE_YARN = 2, LLAMA_ROPE_SCALING_TYPE_LONGROPE = 3 };
enum llama_pooling_type { LLAMA_POOLING_TYPE_UNSPECIFIED = -1, LLAMA_POOLING_TYPE_NONE = 0, LLAMA_POOLING_TYPE_MEAN = 1, LLAMA_POOLING_TYPE_CLS = 2, LLAMA_POOLING_TYPE_LAST = 3, LLAMA_POOLING_TYPE_RANK = 4 };
enum llama_attention_type { LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1, LLAMA_ATTENTION_TYPE_CAUSAL = 0, LLAMA_ATTENTION_TYPE_NON_CAUSAL = 1 };
enum llama_flash_attn_type { LLAMA_FLASH_ATTN_TYPE_AUTO = -1, LLAMA_FLASH_ATTN_TYPE_DISABLED = 0, LLAMA_FLASH_ATTN_TYPE_ENABLED = 1 };
enum ggml_type { GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1, GGML_TYPE_Q4_0 = 2, GGML_TYPE_Q8_0 = 8, GGML_TYPE_BF16 = 30 };

struct llama_model_kv_override; struct llama_model_tensor_buft_override;

struct llama_model_params {
    ggml_backend_dev_t * devices;
    const struct llama_model_tensor_buft_override * tensor_buft_overrides;
    int32_t n_gpu_layers;
    enum llama_split_mode split_mode;
    enum llama_load_mode  load_mode;
    enum llama_lazy_mode lazy_mode;
    int32_t main_gpu;
    const float * tensor_split;
    llama_progress_callback progress_callback;
    void * progress_callback_user_data;
    const struct llama_model_kv_override * kv_overrides;
    bool vocab_only; bool check_tensors; bool use_extra_bufts; bool no_host; bool no_alloc; bool load_mtp;
};

struct llama_sampler_seq_config { llama_seq_id seq_id; struct llama_sampler * sampler; };

struct llama_context_params {
    uint32_t n_ctx; uint32_t n_batch; uint32_t n_ubatch; uint32_t n_seq_max; uint32_t n_rs_seq;
    uint32_t n_outputs_max; uint32_t n_outputs_max_per_seq;
    int32_t n_threads; int32_t n_threads_batch;
    enum llama_context_type ctx_type;
    enum llama_rope_scaling_type rope_scaling_type;
    enum llama_pooling_type pooling_type;
    enum llama_attention_type attention_type;
    enum llama_flash_attn_type flash_attn_type;
    float rope_freq_base; float rope_freq_scale; float yarn_ext_factor; float yarn_attn_factor;
    float yarn_beta_fast; float yarn_beta_slow; uint32_t yarn_orig_ctx; float defrag_thold;
    ggml_backend_sched_eval_callback cb_eval; void * cb_eval_user_data;
    enum ggml_type type_k; enum ggml_type type_v;
    ggml_abort_callback abort_callback; void * abort_callback_data;
    bool embeddings; bool offload_kqv; bool no_perf; bool op_offload; bool swa_full; bool kv_unified;
    struct llama_sampler_seq_config * samplers; size_t n_samplers;
    struct llama_context * ctx_other;
};

typedef struct llama_batch {
    int32_t n_tokens; llama_token * token; float * embd; llama_pos * pos;
    int32_t * n_seq_id; llama_seq_id ** seq_id; int8_t * logits;
} llama_batch;

typedef struct llama_sampler_chain_params { bool no_perf; } llama_sampler_chain_params;
struct llama_perf_context_data { double t_start_ms; double t_load_ms; double t_p_eval_ms; double t_eval_ms; int32_t n_p_eval; int32_t n_eval; int32_t n_reused; };

void llama_backend_init(void);
void llama_backend_free(void);
void llama_log_set(ggml_log_callback log_callback, void * user_data);
struct llama_model_params llama_model_default_params(void);
struct llama_context_params llama_context_default_params(void);
struct llama_model * llama_model_load_from_file(const char * path_model, struct llama_model_params params);
void llama_model_free(struct llama_model * model);
struct llama_context * llama_init_from_model(struct llama_model * model, struct llama_context_params params);
void llama_free(struct llama_context * ctx);
uint32_t llama_n_ctx(const struct llama_context * ctx);
uint32_t llama_n_batch(const struct llama_context * ctx);
llama_memory_t llama_get_memory(const struct llama_context * ctx);
const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);
int32_t llama_model_n_embd(const struct llama_model * model);
int32_t llama_model_n_layer(const struct llama_model * model);
uint64_t llama_model_size(const struct llama_model * model);
uint64_t llama_model_n_params(const struct llama_model * model);
int32_t llama_model_desc(const struct llama_model * model, char * buf, size_t buf_size);
const char * llama_model_chat_template(const struct llama_model * model, const char * name);
int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab);
bool llama_vocab_is_eog(const struct llama_vocab * vocab, llama_token token);
llama_token llama_vocab_bos(const struct llama_vocab * vocab);
llama_token llama_vocab_eos(const struct llama_vocab * vocab);
bool llama_vocab_get_add_bos(const struct llama_vocab * vocab);
bool llama_memory_seq_rm(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1);
void llama_memory_clear(llama_memory_t mem, bool data);
llama_batch llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
void llama_batch_free(llama_batch batch);
int32_t llama_decode(struct llama_context * ctx, llama_batch batch);
void llama_set_n_threads(struct llama_context * ctx, int32_t n_threads, int32_t n_threads_batch);
void llama_synchronize(struct llama_context * ctx);
float * llama_get_logits(struct llama_context * ctx);
float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);
int32_t llama_tokenize(const struct llama_vocab * vocab, const char * text, int32_t text_len, llama_token * tokens, int32_t n_tokens_max, bool add_special, bool parse_special);
int32_t llama_token_to_piece(const struct llama_vocab * vocab, llama_token token, char * buf, int32_t length, int32_t lstrip, bool special);
int32_t llama_detokenize(const struct llama_vocab * vocab, const llama_token * tokens, int32_t n_tokens, char * text, int32_t text_len_max, bool remove_special, bool unparse_special);
llama_sampler_chain_params llama_sampler_chain_default_params(void);
struct llama_sampler * llama_sampler_chain_init(llama_sampler_chain_params params);
void llama_sampler_chain_add(struct llama_sampler * chain, struct llama_sampler * smpl);
struct llama_sampler * llama_sampler_init_greedy(void);
struct llama_sampler * llama_sampler_init_dist(uint32_t seed);
struct llama_sampler * llama_sampler_init_temp(float t);
struct llama_sampler * llama_sampler_init_top_k(int32_t k);
struct llama_sampler * llama_sampler_init_top_p(float p, size_t min_keep);
struct llama_sampler * llama_sampler_init_min_p(float p, size_t min_keep);
llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx);
void llama_sampler_accept(struct llama_sampler * smpl, llama_token token);
void llama_sampler_reset(struct llama_sampler * smpl);
void llama_sampler_free(struct llama_sampler * smpl);
struct llama_perf_context_data llama_perf_context(const struct llama_context * ctx);
void llama_perf_context_reset(struct llama_context * ctx);
int64_t llama_time_us(void);
""")

ggml_ffi = FFI()
ggml_ffi.cdef(r"""
void ggml_backend_load_all(void);
void ggml_backend_load_all_from_path(const char * dir_path);
size_t ggml_backend_dev_count(void);
""")

_lib = None
_ggml = None


def load(bin_dir):
    """Load llama.dll from a release folder, then ask ggml.dll to scan that folder for backend DLLs
    (ggml-cpu-*.dll, ggml-cuda.dll ...). llama-cli does the same through ggml_backend_load_all()."""
    global _lib, _ggml
    if _lib is not None:
        return _lib
    bin_dir = os.path.abspath(bin_dir)
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(bin_dir)
    os.environ['PATH'] = bin_dir + os.pathsep + os.environ.get('PATH', '')
    _ggml = ggml_ffi.dlopen(os.path.join(bin_dir, 'ggml.dll'))
    _ggml.ggml_backend_load_all_from_path(bin_dir.encode())
    _lib = ffi.dlopen(os.path.join(bin_dir, 'llama.dll'))
    return _lib
