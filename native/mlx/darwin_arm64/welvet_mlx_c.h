#pragma once

/* C ABI for welvet.mlx — Go FFI / Welvet nativemlx package.
 * Forward-only Apple MLX GPU. Host buffers are float32; `dtype` selects MLX compute type.
 * No CPU numerical fallback. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  if defined(WELVET_MLX_BUILD_SHARED)
#    define WMLX_API __declspec(dllexport)
#  elif defined(WMLX_DLLIMPORT)
#    define WMLX_API __declspec(dllimport)
#  else
#    define WMLX_API
#  endif
#else
#  define WMLX_API __attribute__((visibility("default")))
#endif

typedef struct {
  int32_t code; /* 0=ok, 1=unavailable, 2=failed, 3=unsupported */
  char message[256];
} wmlx_status;

typedef struct wmlx_session wmlx_session;
typedef struct wmlx_sequential wmlx_sequential;

WMLX_API const char* wmlx_version(void);
WMLX_API int32_t wmlx_has_mlx(void);
WMLX_API int32_t wmlx_layer_supported(const char* layer, const char* dtype);
WMLX_API const char* const* wmlx_supported_layers(void);
WMLX_API wmlx_status wmlx_detect(int32_t* mlx_found, int32_t* metal_ok, int32_t* gpu_ok,
                                 char* mlx_ver, int32_t mlx_ver_n, char* device_name,
                                 int32_t device_name_n);

WMLX_API wmlx_status wmlx_session_open_gpu(wmlx_session** out);
WMLX_API void wmlx_session_close(wmlx_session* s);
WMLX_API int32_t wmlx_session_ready(const wmlx_session* s);

/* dtype: "float32"|"float16"|"bfloat16"|"float64"|
 *        "int8"|"int16"|"int32"|"int64"|
 *        "uint8"|"uint16"|"uint32"|"uint64"|"complex64" (NULL → float32).
 * Host x/W/y remain float32; library casts to MLX compute dtype on GPU.
 * float64 is rejected on GPU (MLX CPU-only). Caps gate per layer. */

WMLX_API wmlx_status wmlx_dense_prepare_f32(wmlx_session* s, int32_t batch, int32_t in, int32_t out,
                                            const float* W, const float* bias, const char* dtype);
WMLX_API wmlx_status wmlx_dense_execute_f32(wmlx_session* s, const float* x, float* y);
WMLX_API wmlx_status wmlx_dense_forward_f32(wmlx_session* s, int32_t batch, int32_t in, int32_t out,
                                            const float* W, const float* bias, const float* x,
                                            float* y, const char* dtype);

WMLX_API wmlx_status wmlx_softmax_forward_f32(wmlx_session* s, int32_t rows, int32_t cols,
                                              const float* x, float* y, const char* dtype);
WMLX_API wmlx_status wmlx_rmsnorm_forward_f32(wmlx_session* s, int32_t batch, int32_t seq,
                                              int32_t dim, float eps, const float* gamma,
                                              const float* x, float* y, const char* dtype);
WMLX_API wmlx_status wmlx_layernorm_forward_f32(wmlx_session* s, int32_t batch, int32_t seq,
                                                int32_t dim, float eps, const float* gamma,
                                                const float* beta, const float* x, float* y,
                                                const char* dtype);

WMLX_API wmlx_status wmlx_seq_create(wmlx_sequential** out);
WMLX_API void wmlx_seq_free(wmlx_sequential* seq);
WMLX_API wmlx_status wmlx_seq_push_dense(wmlx_sequential* seq, int32_t in, int32_t out,
                                         const float* W, const float* bias);
WMLX_API wmlx_status wmlx_seq_push_rmsnorm(wmlx_sequential* seq, int32_t dim, float eps,
                                           const float* gamma);
WMLX_API wmlx_status wmlx_seq_push_layernorm(wmlx_sequential* seq, int32_t dim, float eps,
                                             const float* gamma, const float* beta);
WMLX_API wmlx_status wmlx_seq_push_softmax(wmlx_sequential* seq, int32_t cols);
WMLX_API int32_t wmlx_seq_in_features(const wmlx_sequential* seq);
WMLX_API int32_t wmlx_seq_out_features(const wmlx_sequential* seq);
WMLX_API int32_t wmlx_seq_size(const wmlx_sequential* seq);
WMLX_API wmlx_status wmlx_seq_forward_f32(wmlx_session* s, wmlx_sequential* seq, int32_t batch,
                                          int32_t seq_len, const float* x, float* y);

WMLX_API void wmlx_last_phase_ns(const wmlx_session* s, int64_t* create_ns, int64_t* finalize_ns,
                                 int64_t* execute_ns);

WMLX_API wmlx_status wmlx_cnn2_forward_f32(wmlx_session* s, int32_t n, int32_t h, int32_t w,
                                           int32_t cin, int32_t cout, const float* W,
                                           const float* bias, const float* x, float* y,
                                           const char* dtype);
WMLX_API wmlx_status wmlx_cnn3_forward_f32(wmlx_session* s, int32_t n, int32_t d, int32_t h,
                                           int32_t w, int32_t cin, int32_t cout, const float* W,
                                           const float* bias, const float* x, float* y,
                                           const char* dtype);
WMLX_API wmlx_status wmlx_convt2_forward_f32(wmlx_session* s, int32_t n, int32_t h, int32_t w,
                                             int32_t cin, int32_t cout, const float* W,
                                             const float* bias, const float* x, float* y,
                                             const char* dtype);
WMLX_API wmlx_status wmlx_convt3_forward_f32(wmlx_session* s, int32_t n, int32_t d, int32_t h,
                                             int32_t w, int32_t cin, int32_t cout, const float* W,
                                             const float* bias, const float* x, float* y,
                                             const char* dtype);
WMLX_API wmlx_status wmlx_lstm_forward_f32(wmlx_session* s, int32_t seq, int32_t batch,
                                           int32_t input_size, int32_t hidden_size, const float* w_in,
                                           const float* w_rec, const float* b, const float* x,
                                           float* y, const char* dtype);
WMLX_API wmlx_status wmlx_embedding_forward_f32(wmlx_session* s, int32_t vocab, int32_t dim,
                                                int32_t n, const float* table, const int32_t* ids,
                                                float* y, const char* dtype);
WMLX_API wmlx_status wmlx_residual_forward_f32(wmlx_session* s, int32_t n, const float* a,
                                               const float* b, float* y, const char* dtype);
WMLX_API wmlx_status wmlx_swiglu_forward_f32(wmlx_session* s, int32_t batch, int32_t in,
                                             int32_t hidden, const float* w_gate, const float* w_up,
                                             const float* w_down, const float* x, float* y,
                                             const char* dtype);
WMLX_API wmlx_status wmlx_mha_forward_f32(wmlx_session* s, int32_t batch, int32_t seq, int32_t dim,
                                          int32_t heads, const float* w_q, const float* w_k,
                                          const float* w_v, const float* w_o, const float* x,
                                          float* y, const char* dtype);

#ifdef __cplusplus
}
#endif
