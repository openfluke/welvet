#pragma once

/* C ABI for welvet.qnn — Go FFI / Welvet nativeqnn package.
 * Forward-only Hexagon NPU (QNN HTP). User syncs weights from Go. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  if defined(WELVET_QNN_BUILD_SHARED)
#    define WQNN_API __declspec(dllexport)
#  elif defined(WQNN_DLLIMPORT)
#    define WQNN_API __declspec(dllimport)
#  else
#    define WQNN_API
#  endif
#else
#  define WQNN_API __attribute__((visibility("default")))
#endif

typedef struct {
  int32_t code; /* 0=ok, 1=unavailable, 2=failed, 3=unsupported */
  char message[256];
} wqnn_status;

typedef struct wqnn_session wqnn_session;
typedef struct wqnn_sequential wqnn_sequential;

WQNN_API const char* wqnn_version(void);

/* 1 if this binary was built with QAIRT headers. */
WQNN_API int32_t wqnn_has_sdk(void);

/* Caps: 1 if (layer, dtype) is allowed on HTP for this library build/host policy. */
WQNN_API int32_t wqnn_layer_supported(const char* layer, const char* dtype);

/* Null-terminated list of supported layer names (static storage). */
WQNN_API const char* const* wqnn_supported_layers(void);

/* Detect QNN/HTP stack (no backendCreate). fills buffers; returns status. */
WQNN_API wqnn_status wqnn_detect(int32_t* sdk_found, int32_t* htp_lib, int32_t* htp_prov,
                                 char* sdk_root, int32_t sdk_root_n, char* build_id,
                                 int32_t build_id_n);

/* Open HTP (NPU) session. Caller must wqnn_session_close. */
WQNN_API wqnn_status wqnn_session_open_htp(wqnn_session** out);
WQNN_API void wqnn_session_close(wqnn_session* s);
WQNN_API int32_t wqnn_session_ready(const wqnn_session* s);

/* Dense: prepare bakes W into an HTP graph; execute reuses it (timing = NPU run only).
 * forward = prepare-if-needed + execute (weights ignored on cache hit for same batch/in/out). */
WQNN_API wqnn_status wqnn_dense_prepare_f32(wqnn_session* s, int32_t batch, int32_t in, int32_t out,
                                            const float* W, const float* bias);
WQNN_API wqnn_status wqnn_dense_execute_f32(wqnn_session* s, const float* x, float* y);
WQNN_API wqnn_status wqnn_dense_forward_f32(wqnn_session* s, int32_t batch, int32_t in, int32_t out,
                                            const float* W, const float* bias, const float* x,
                                            float* y);

WQNN_API wqnn_status wqnn_softmax_forward_f32(wqnn_session* s, int32_t rows, int32_t cols,
                                              const float* x, float* y);

WQNN_API wqnn_status wqnn_rmsnorm_forward_f32(wqnn_session* s, int32_t batch, int32_t seq,
                                              int32_t dim, float eps, const float* gamma,
                                              const float* x, float* y);

WQNN_API wqnn_status wqnn_layernorm_forward_f32(wqnn_session* s, int32_t batch, int32_t seq,
                                                int32_t dim, float eps, const float* gamma,
                                                const float* beta, const float* x, float* y);

/* Sequential (Welvet-style stack). push_* reject shape mismatch / unsupported. */
WQNN_API wqnn_status wqnn_seq_create(wqnn_sequential** out);
WQNN_API void wqnn_seq_free(wqnn_sequential* seq);
WQNN_API wqnn_status wqnn_seq_push_dense(wqnn_sequential* seq, int32_t in, int32_t out,
                                         const float* W, const float* bias);
WQNN_API wqnn_status wqnn_seq_push_rmsnorm(wqnn_sequential* seq, int32_t dim, float eps,
                                           const float* gamma);
WQNN_API wqnn_status wqnn_seq_push_layernorm(wqnn_sequential* seq, int32_t dim, float eps,
                                             const float* gamma, const float* beta);
WQNN_API wqnn_status wqnn_seq_push_softmax(wqnn_sequential* seq, int32_t cols);
WQNN_API int32_t wqnn_seq_in_features(const wqnn_sequential* seq);
WQNN_API int32_t wqnn_seq_out_features(const wqnn_sequential* seq);
WQNN_API int32_t wqnn_seq_size(const wqnn_sequential* seq);
WQNN_API wqnn_status wqnn_seq_forward_f32(wqnn_session* s, wqnn_sequential* seq, int32_t batch,
                                          int32_t seq_len, const float* x, float* y);

/* Last graph phase costs from the most recent forward/prepare/execute on this session (ns). */
WQNN_API void wqnn_last_phase_ns(const wqnn_session* s, int64_t* create_ns, int64_t* finalize_ns,
                                 int64_t* execute_ns);

/* Extra HTP layers (README caps table) — f32 host API. */
WQNN_API wqnn_status wqnn_cnn2_forward_f32(wqnn_session* s, int32_t n, int32_t h, int32_t w,
                                           int32_t cin, int32_t cout, const float* W,
                                           const float* bias, const float* x, float* y);
WQNN_API wqnn_status wqnn_cnn3_forward_f32(wqnn_session* s, int32_t n, int32_t d, int32_t h,
                                           int32_t w, int32_t cin, int32_t cout, const float* W,
                                           const float* bias, const float* x, float* y);
WQNN_API wqnn_status wqnn_convt2_forward_f32(wqnn_session* s, int32_t n, int32_t h, int32_t w,
                                             int32_t cin, int32_t cout, const float* W,
                                             const float* bias, const float* x, float* y);
WQNN_API wqnn_status wqnn_convt3_forward_f32(wqnn_session* s, int32_t n, int32_t d, int32_t h,
                                             int32_t w, int32_t cin, int32_t cout, const float* W,
                                             const float* bias, const float* x, float* y);
WQNN_API wqnn_status wqnn_lstm_forward_f32(wqnn_session* s, int32_t seq, int32_t batch,
                                           int32_t input_size, int32_t hidden_size, const float* w_in,
                                           const float* w_rec, const float* b, const float* x,
                                           float* y);
WQNN_API wqnn_status wqnn_embedding_forward_f32(wqnn_session* s, int32_t vocab, int32_t dim,
                                                int32_t n, const float* table, const int32_t* ids,
                                                float* y);
WQNN_API wqnn_status wqnn_residual_forward_f32(wqnn_session* s, int32_t n, const float* a,
                                               const float* b, float* y);
WQNN_API wqnn_status wqnn_swiglu_forward_f32(wqnn_session* s, int32_t batch, int32_t in,
                                             int32_t hidden, const float* w_gate, const float* w_up,
                                             const float* w_down, const float* x, float* y);
WQNN_API wqnn_status wqnn_mha_forward_f32(wqnn_session* s, int32_t batch, int32_t seq, int32_t dim,
                                          int32_t heads, const float* w_q, const float* w_k,
                                          const float* w_v, const float* w_o, const float* x,
                                          float* y);

#ifdef __cplusplus
}
#endif
