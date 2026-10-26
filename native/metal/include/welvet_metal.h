#pragma once

/* Stable C ABI for welvet.metal — used by Go nativemetal and w2a go_v_metal. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(WELVET_BUILD_SHARED)
#  ifdef WELVET_EXPORTS
#    define WELVET_API __declspec(dllexport)
#  else
#    define WELVET_API __declspec(dllimport)
#  endif
#else
#  define WELVET_API __attribute__((visibility("default")))
#endif

typedef struct welvet_status {
  int32_t code; /* 0=ok, 1=invalid, 2=unsupported, 3=gap, 4=internal */
  char message[256];
} welvet_status;

/* Engine identity — version follows Go-mirrored board (0.{round(earned)}). */
WELVET_API const char* welvet_version(void);
WELVET_API const char* welvet_runtime_identity(void); /* Metal device + build */

WELVET_API welvet_status welvet_metal_init(void);
WELVET_API int32_t welvet_metal_available(void);
WELVET_API int32_t welvet_gaps_count(void);
WELVET_API int32_t welvet_gaps_done(void);

/* Versioned persistent model sessions. All passed buffers are copied during the
 * call; no Go pointers are retained. Native weights remain authoritative until
 * explicitly exported. Release sessions and output blobs with their own APIs.
 * Model descriptor WMN1 and snapshot NWS1 are defined in docs/GO_INTEGRATION.md. */
typedef struct welvet_model_session welvet_model_session;
typedef struct welvet_metal_blob { uint8_t* data; size_t size; } welvet_metal_blob;
WELVET_API int32_t welvet_model_abi_version(void);
WELVET_API welvet_status welvet_model_mount(const uint8_t*, size_t, welvet_model_session**);
WELVET_API welvet_status welvet_model_forward(welvet_model_session*, const uint8_t*, size_t, welvet_metal_blob*);
WELVET_API welvet_status welvet_model_train_mse(welvet_model_session*, int32_t mode, double lr, int32_t ticks, const uint8_t*, size_t, double* loss);
WELVET_API welvet_status welvet_model_export_weights(welvet_model_session*, welvet_metal_blob*);
WELVET_API welvet_status welvet_model_sync(welvet_model_session*, int32_t event);
WELVET_API void welvet_model_release(welvet_model_session*);
WELVET_API void welvet_metal_blob_free(welvet_metal_blob*);

/* One-shot Dense Metal forward (mount+execute+release internally). */
WELVET_API welvet_status welvet_dense_forward_f32(
    int32_t batch, int32_t in, int32_t out, int32_t act, int32_t dtype, int32_t backend,
    const float* W_init, const double* bias, size_t bias_len, const float* x, float* pre,
    float* post);

/* Dense NormalBP leaf: Metal forward → (MSEGrad from target | use grad_out) → Metal backward → optional SGD.
 * use_target: 0 = grad_or_target is gradOut; 1 = target (MSE).
 * lr == 0 skips ApplyGradSGD (W_after == W decoded).
 * Outputs: pre/post [batch*out], dX [batch*in], dW [out*in], W_after [out*in], *loss (0 if !use_target).
 */
WELVET_API welvet_status welvet_dense_normalbp_f32(
    int32_t batch, int32_t in, int32_t out, int32_t act, int32_t dtype, int32_t backend,
    const float* W_init, const double* bias, size_t bias_len, const float* x,
    const float* grad_or_target, int32_t use_target, double lr, float* pre, float* post, float* dX,
    float* dW, float* W_after, double* loss);

/* Resident session: mount once, execute many (w2a mount vs exec timing). */
typedef struct welvet_dense_session welvet_dense_session;

WELVET_API welvet_status welvet_dense_mount(
    int32_t batch, int32_t in, int32_t out, int32_t act, int32_t dtype,
    const float* W_init, const double* bias, size_t bias_len, welvet_dense_session** sess);

WELVET_API welvet_status welvet_dense_execute(welvet_dense_session* s, const float* x, float* pre,
                                              float* post);

WELVET_API void welvet_dense_release(welvet_dense_session* s);

/* MHA ModeSelf forward — WVM1 wire. Q/K/V/O Dense on Metal; Metal attention kernels. */
WELVET_API welvet_status welvet_mha_forward_f32(
    int32_t batch, int32_t seq, int32_t d_model, int32_t num_heads, int32_t num_kv_heads,
    int32_t head_dim, int32_t max_seq_len, int32_t mask, int32_t pos, int32_t dtype, int32_t backend,
    const float* Q, const float* K, const float* V, const float* O, const float* x, float* pre,
    float* post);

/* MHA NormalBP leaf (ModeSelf): forward → (MSEGrad | grad_out) → backward → optional SGD.
 * use_target: 0 = grad_or_target is gradOut; 1 = target (MSE).
 * Outputs: pre [batch*seq*q_dim], post/dX [batch*seq*d], dW/W_after = concat(Q,K,V,O), *loss.
 */
WELVET_API welvet_status welvet_mha_normalbp_f32(
    int32_t batch, int32_t seq, int32_t d_model, int32_t num_heads, int32_t num_kv_heads,
    int32_t head_dim, int32_t max_seq_len, int32_t mask, int32_t pos, int32_t dtype, int32_t backend,
    const float* Q, const float* K, const float* V, const float* O, const float* x,
    const float* grad_or_target, int32_t use_target, double lr, float* pre, float* post, float* dX,
    float* dW, float* W_after, double* loss);

/* SwiGLU forward — WVS1 wire. Gate/Up/Down Dense on Metal; Metal SiLU/product kernels. */
WELVET_API welvet_status welvet_swiglu_forward_f32(
    int32_t batch, int32_t seq, int32_t input_dim, int32_t intermediate_dim, int32_t dtype,
    int32_t backend, const float* Gate, const float* Up, const float* Down, const float* x,
    float* pre, float* post);

/* SwiGLU NormalBP leaf: forward → (MSEGrad | grad_out) → backward → optional SGD.
 * use_target: 0 = grad_or_target is gradOut; 1 = target (MSE).
 * Outputs: pre [batch*seq*inter], post/dX [batch*seq*in], dW/W_after = concat(Gate,Up,Down), *loss.
 */
WELVET_API welvet_status welvet_swiglu_normalbp_f32(
    int32_t batch, int32_t seq, int32_t input_dim, int32_t intermediate_dim, int32_t dtype,
    int32_t backend, const float* Gate, const float* Up, const float* Down, const float* x,
    const float* grad_or_target, int32_t use_target, double lr, float* pre, float* post, float* dX,
    float* dW, float* W_after, double* loss);

/* RMSNorm forward — WVR1 wire. γ FormatNone×34 + Metal RMS kernels. */
WELVET_API welvet_status welvet_rmsnorm_forward_f32(
    int32_t batch, int32_t seq, int32_t dim, double eps, int32_t dtype, int32_t backend,
    const float* gamma, const float* x, float* pre, float* post);

/* RMSNorm NormalBP leaf: forward → (MSEGrad | grad_out) → backward → optional SGD.
 * use_target: 0 = grad_or_target is gradOut; 1 = target (MSE).
 * Outputs: pre/post/dX [batch*seq*dim], dW/gamma_after [dim], *loss.
 */
WELVET_API welvet_status welvet_rmsnorm_normalbp_f32(
    int32_t batch, int32_t seq, int32_t dim, double eps, int32_t dtype, int32_t backend,
    const float* gamma_init, const float* x, const float* grad_or_target, int32_t use_target,
    double lr, float* pre, float* post, float* dX, float* dW, float* gamma_after, double* loss);

/* LayerNorm forward — WVL1 wire. γ/β FormatNone×34 + Metal mean/variance kernels. */
WELVET_API welvet_status welvet_layernorm_forward_f32(
    int32_t batch, int32_t seq, int32_t dim, double eps, int32_t dtype, int32_t backend,
    const float* gamma, const float* beta, const float* x, float* pre, float* post);

/* LayerNorm NormalBP leaf: forward → (MSEGrad | grad_out) → backward → optional SGD.
 * use_target: 0 = grad_or_target is gradOut; 1 = target (MSE).
 * Outputs: pre/post/dX [batch*seq*dim], dW [2*dim]=dγ|dβ, gamma_after/beta_after [dim], *loss.
 */
WELVET_API welvet_status welvet_layernorm_normalbp_f32(
    int32_t batch, int32_t seq, int32_t dim, double eps, int32_t dtype, int32_t backend,
    const float* gamma_init, const float* beta_init, const float* x, const float* grad_or_target,
    int32_t use_target, double lr, float* pre, float* post, float* dX, float* dW,
    float* gamma_after, float* beta_after, double* loss);

/* Softmax forward — WSM1 wire. All Kind variants; Metal stable Softmax. */
WELVET_API welvet_status welvet_softmax_forward_f32(
    int32_t n, int32_t dim, int32_t seq_len, int32_t kind, int32_t rows, int32_t cols,
    double temperature, double entmax_alpha, int32_t gumbel_seed, int32_t dtype, int32_t backend,
    int32_t rank, const int32_t* shape, const uint8_t* mask, int32_t mask_len, const int32_t* hier,
    int32_t hier_len, const float* x, float* pre, float* post);

/* Softmax NormalBP leaf (weightless dX): forward → (MSEGrad | grad_out) → backward → optional SGD.
 * use_target: 0 = grad_or_target is gradOut; 1 = target (MSE). lr applied as no-op SGD.
 * Outputs: pre/post/dX [n], *loss. dW always empty.
 */
WELVET_API welvet_status welvet_softmax_normalbp_f32(
    int32_t n, int32_t dim, int32_t seq_len, int32_t kind, int32_t rows, int32_t cols,
    double temperature, double entmax_alpha, int32_t gumbel_seed, int32_t dtype, int32_t backend,
    int32_t rank, const int32_t* shape, const uint8_t* mask, int32_t mask_len, const int32_t* hier,
    int32_t hier_len, const float* x, const float* grad_or_target, int32_t use_target, double lr,
    float* pre, float* post, float* dX, double* loss);

/* Embedding gather — WEM1 wire. Vocab×Emb FormatNone×34 decode_row_f64 (Go/cpp exact_bit). */
WELVET_API welvet_status welvet_embedding_forward_f32(
    int32_t batch, int32_t vocab, int32_t emb, int32_t seq_len, int32_t in_rank, int32_t in_last,
    int32_t act, int32_t dtype, int32_t backend, const float* W, const float* x, float* pre,
    float* post);

/* Embedding NormalBP leaf: forward → (MSEGrad | grad_out) → scatter dW; dX zeros; optional SGD.
 * use_target: 0 = grad_or_target is gradOut; 1 = target (MSE).
 * Outputs: pre/post [out_n], dX [x_n] zeros, dW/W_after [vocab*emb], *loss.
 */
WELVET_API welvet_status welvet_embedding_normalbp_f32(
    int32_t batch, int32_t vocab, int32_t emb, int32_t seq_len, int32_t in_rank, int32_t in_last,
    int32_t act, int32_t dtype, int32_t backend, const float* W_init, const float* x,
    const float* grad_or_target, int32_t use_target, double lr, float* pre, float* post, float* dX,
    float* dW, float* W_after, double* loss);

/* Sequential Dense chain — WSQ1 wire. packed: Depth×[Dim×Dim], x: [batch*seq*dim]
 * (seq=1 → [batch*dim]). pre/post: same layout (last child). Chains Metal soft-double Dense. */
WELVET_API welvet_status welvet_sequential_forward_f32(
    int32_t batch, int32_t seq, int32_t dim, int32_t depth, int32_t child_act, int32_t dtype,
    int32_t backend, const float* packed, const float* x, float* pre, float* post);

/* Sequential NormalBP leaf. use_target: 0=gradOut on post, 1=MSE target. lr==0 skips SGD.
 * dW/W_after: loom Depth×[Dim×Dim]. */
WELVET_API welvet_status welvet_sequential_normalbp_f32(
    int32_t batch, int32_t seq, int32_t dim, int32_t depth, int32_t child_act, int32_t dtype,
    int32_t backend, const float* packed, const float* x, const float* grad_or_target,
    int32_t use_target, double lr, float* pre, float* post, float* dX, float* dW, float* W_after,
    double* loss);

/* Residual y=F(x)+x — WRE1 wire. packed: Depth×[Dim×Dim] Dense F, x: [batch*seq*dim]
 * (seq=1 → [batch*dim]). pre: last F child pre; post: skip-add. F via Metal soft-double Dense. */
WELVET_API welvet_status welvet_residual_forward_f32(
    int32_t batch, int32_t seq, int32_t dim, int32_t depth, int32_t child_act, int32_t dtype,
    int32_t backend, const float* packed, const float* x, float* pre, float* post);

/* Residual NormalBP leaf. use_target: 0=gradOut on post, 1=MSE target. lr==0 skips SGD.
 * dW/W_after: loom Depth×[Dim×Dim] (F children only). */
WELVET_API welvet_status welvet_residual_normalbp_f32(
    int32_t batch, int32_t seq, int32_t dim, int32_t depth, int32_t child_act, int32_t dtype,
    int32_t backend, const float* packed, const float* x, const float* grad_or_target,
    int32_t use_target, double lr, float* pre, float* post, float* dX, float* dW, float* W_after,
    double* loss);

/* seqmix Kind name — WSX1 smoke. kind: 0=attention 1=ssm 2=linear_attn 3=conv_mix.
 * Writes NUL-terminated name into buf (max buf_len). */
WELVET_API welvet_status welvet_seqmix_kind_name(int32_t kind, char* buf, int32_t buf_len);

/* Mixer Kind tags: mha→0, mamba→1, gdn→2 (matches seqmix.Kind iota). */
WELVET_API int32_t welvet_seqmix_mha_kind(void);
WELVET_API int32_t welvet_seqmix_mamba_kind(void);
WELVET_API int32_t welvet_seqmix_gdn_kind(void);

/* CNN1/2/3 — WVC1/2/3 wire. Metal im2col → Metal soft-double Dense → loom. */
WELVET_API welvet_status welvet_cnn1_forward_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t seq_len, int32_t kernel, int32_t stride,
    int32_t padding, int32_t act, int32_t dtype, int32_t backend, const float* W, const double* bias,
    size_t bias_len, const float* x, float* pre, float* post);

WELVET_API welvet_status welvet_cnn2_forward_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t height, int32_t width, int32_t kernel,
    int32_t stride, int32_t padding, int32_t act, int32_t dtype, int32_t backend, const float* W,
    const double* bias, size_t bias_len, const float* x, float* pre, float* post);

WELVET_API welvet_status welvet_cnn3_forward_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t depth, int32_t height, int32_t width,
    int32_t kernel, int32_t stride, int32_t padding, int32_t act, int32_t dtype, int32_t backend,
    const float* W, const double* bias, size_t bias_len, const float* x, float* pre, float* post);

/* CNN1/2/3 NormalBP leaf: forward → (MSEGrad | grad_out) → backward → optional SGD.
 * use_target: 0 = grad_or_target is gradOut; 1 = target (MSE).
 * dW/W_after: [filters * patch_dim].
 */
WELVET_API welvet_status welvet_cnn1_normalbp_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t seq_len, int32_t kernel, int32_t stride,
    int32_t padding, int32_t act, int32_t dtype, int32_t backend, const float* W,
    const double* bias, size_t bias_len, const float* x, const float* grad_or_target,
    int32_t use_target, double lr, float* pre, float* post, float* dX, float* dW, float* W_after,
    double* loss);

WELVET_API welvet_status welvet_cnn2_normalbp_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t height, int32_t width, int32_t kernel,
    int32_t stride, int32_t padding, int32_t act, int32_t dtype, int32_t backend, const float* W,
    const double* bias, size_t bias_len, const float* x, const float* grad_or_target,
    int32_t use_target, double lr, float* pre, float* post, float* dX, float* dW, float* W_after,
    double* loss);

WELVET_API welvet_status welvet_cnn3_normalbp_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t depth, int32_t height, int32_t width,
    int32_t kernel, int32_t stride, int32_t padding, int32_t act, int32_t dtype, int32_t backend,
    const float* W, const double* bias, size_t bias_len, const float* x, const float* grad_or_target,
    int32_t use_target, double lr, float* pre, float* post, float* dX, float* dW, float* W_after,
    double* loss);

/* ConvT1/2/3 — WVT1/2/3 wire. Metal scatter and activation. No bias. */
WELVET_API welvet_status welvet_convt1_forward_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t seq_len, int32_t kernel, int32_t stride,
    int32_t padding, int32_t output_padding, int32_t act, int32_t dtype, int32_t backend,
    const float* W, const float* x, float* pre, float* post);

WELVET_API welvet_status welvet_convt2_forward_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t height, int32_t width, int32_t kernel,
    int32_t stride, int32_t padding, int32_t output_padding, int32_t act, int32_t dtype,
    int32_t backend, const float* W, const float* x, float* pre, float* post);

WELVET_API welvet_status welvet_convt3_forward_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t depth, int32_t height, int32_t width,
    int32_t kernel, int32_t stride, int32_t padding, int32_t output_padding, int32_t act,
    int32_t dtype, int32_t backend, const float* W, const float* x, float* pre, float* post);


/* ConvT1/2/3 NormalBP leaf: scatter fwd + act′ + scatter adjoint + optional SGD. No bias.
 * use_target: 0 = grad_or_target is gradOut; 1 = target (MSE).
 */
WELVET_API welvet_status welvet_convt1_normalbp_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t seq_len, int32_t kernel, int32_t stride,
    int32_t padding, int32_t output_padding, int32_t act, int32_t dtype, int32_t backend,
    const float* W, const float* x, const float* grad_or_target, int32_t use_target, double lr,
    float* pre, float* post, float* dX, float* dW, float* W_after, double* loss);

WELVET_API welvet_status welvet_convt2_normalbp_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t height, int32_t width, int32_t kernel,
    int32_t stride, int32_t padding, int32_t output_padding, int32_t act, int32_t dtype,
    int32_t backend, const float* W, const float* x, const float* grad_or_target, int32_t use_target,
    double lr, float* pre, float* post, float* dX, float* dW, float* W_after, double* loss);

WELVET_API welvet_status welvet_convt3_normalbp_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t depth, int32_t height, int32_t width,
    int32_t kernel, int32_t stride, int32_t padding, int32_t output_padding, int32_t act,
    int32_t dtype, int32_t backend, const float* W, const float* x, const float* grad_or_target,
    int32_t use_target, double lr, float* pre, float* post, float* dX, float* dW, float* W_after,
    double* loss);

/* RNN — WRN1 wire. packed [IH|HH|bias]; Dense soft-double + Metal tanh. */
WELVET_API welvet_status welvet_rnn_forward_f32(
    int32_t batch, int32_t input_size, int32_t hidden_size, int32_t seq_len, int32_t dtype,
    int32_t backend, const float* packed, const float* x, float* pre, float* post);

/* RNN NormalBP leaf. use_target: 0=gradOut on post, 1=MSE target. lr==0 skips SGD.
 * dW/W_after: loom pack [IH|HH|bias]. */
WELVET_API welvet_status welvet_rnn_normalbp_f32(
    int32_t batch, int32_t input_size, int32_t hidden_size, int32_t seq_len, int32_t dtype,
    int32_t backend, const float* packed, const float* x, const float* grad_or_target,
    int32_t use_target, double lr, float* pre, float* post, float* dX, float* dW, float* W_after,
    double* loss);

/* LSTM — WLS1 wire. packed 4×[IH|HH|bias]; Dense soft-double + Metal gate kernels. */
WELVET_API welvet_status welvet_lstm_forward_f32(
    int32_t batch, int32_t input_size, int32_t hidden_size, int32_t seq_len, int32_t dtype,
    int32_t backend, const float* packed, const float* x, float* pre, float* post);

/* LSTM NormalBP leaf. use_target: 0=gradOut on post, 1=MSE target. lr==0 skips SGD.
 * dW/W_after: loom pack 4×[IH|HH|bias]. pre=[batch,seq,5H]. */
WELVET_API welvet_status welvet_lstm_normalbp_f32(
    int32_t batch, int32_t input_size, int32_t hidden_size, int32_t seq_len, int32_t dtype,
    int32_t backend, const float* packed, const float* x, const float* grad_or_target,
    int32_t use_target, double lr, float* pre, float* post, float* dX, float* dW, float* W_after,
    double* loss);

/* GDN (Gated DeltaNet) — WGD1 wire. Float32 FormatNone Metal kernels.
 * packed: InQKV|InZ|InB|InA|Out|Conv|ALog|DtBias|γ. dtype ignored. */
WELVET_API welvet_status welvet_gdn_forward_f32(
    int32_t batch, int32_t seq, int32_t hidden, int32_t num_key_heads, int32_t num_value_heads,
    int32_t key_head_dim, int32_t value_head_dim, int32_t conv_kernel, int32_t dtype, int32_t backend,
    const float* packed, const float* x, float* pre, float* post);

/* GDN truncated-BPTT NormalBP leaf (f32). use_target: 0=gradOut on post, 1=MSE target. lr==0 skips SGD. */
WELVET_API welvet_status welvet_gdn_normalbp_f32(
    int32_t batch, int32_t seq, int32_t hidden, int32_t num_key_heads, int32_t num_value_heads,
    int32_t key_head_dim, int32_t value_head_dim, int32_t conv_kernel, int32_t dtype, int32_t backend,
    const float* packed, const float* x, const float* grad_or_target, int32_t use_target, double lr,
    float* pre, float* post, float* dX, float* dW, float* W_after, double* loss);

/* Stack NestedStack-of-Dense Dim→Dim NormalBP leaf (STKB/STKO). */
WELVET_API welvet_status welvet_stack_normalbp_f32(
    int32_t batch, int32_t seq, int32_t dim, int32_t n_children, int32_t child_act, int32_t dtype,
    int32_t backend, const float* packed, const float* x, const float* grad_or_target,
    int32_t use_target, double lr, float* pre, float* post, float* dX, float* dW, float* W_after,
    double* loss);

/* Mamba — WMB1 wire. Dense soft-double In/Out FormatNone×34 + Metal selective scan.
 * packed: InProj|OutProj|ALog|DSkip. */
WELVET_API welvet_status welvet_mamba_forward_f32(
    int32_t batch, int32_t seq, int32_t d_model, int32_t d_state, int32_t expand, int32_t dtype,
    int32_t backend, const float* packed, const float* x, float* pre, float* post);

/* Mamba NormalBP leaf. use_target: 0=gradOut on post, 1=MSE target. lr==0 skips SGD.
 * packed/dW/W_after: InProj|OutProj|ALog|DSkip. */
WELVET_API welvet_status welvet_mamba_normalbp_f32(
    int32_t batch, int32_t seq, int32_t d_model, int32_t d_state, int32_t expand, int32_t dtype,
    int32_t backend, const float* packed, const float* x, const float* grad_or_target,
    int32_t use_target, double lr, float* pre, float* post, float* dX, float* dW, float* W_after,
    double* loss);

/* KMeans soft assignment — WKM1 wire. Centers FormatNone×34, Metal soft assignment kernels.
 * output_mode: 0=probabilities (K), 1=features (D). */
WELVET_API welvet_status welvet_kmeans_forward_f32(
    int32_t n, int32_t feature_dim, int32_t num_clusters, int32_t output_mode, int32_t act,
    double temperature, int32_t dtype, int32_t backend, int32_t rank, const int32_t* shape,
    const float* W, const float* x, float* pre, float* post);

/* Metal NormalBP; use_target=1 computes MSE; otherwise consumes gradOut.
 * dW/W_after use the packed forward weight layout; dX uses input layout. */
WELVET_API welvet_status welvet_kmeans_normalbp_f32(
    int32_t batch, int32_t seq, int32_t num_clusters, int32_t feature_dim, int32_t output_mode,
    int32_t act, double temperature, int32_t dtype, int32_t backend, const float* W, const float* x,
    const float* grad_or_target, int32_t use_target, double lr, float* pre, float* post, float* dX,
    float* dW, float* W_after, double* loss);

/* Parallel Dense branches — WPL1 wire. Soft-double Dense FormatNone×34 + Metal combine/routing.
 * combine: 0 concat,1 add,2 avg,3 filter,4 max,5 sparsek,6 disagree.
 * packed: Branches×[OutFeat×Dim] (+ Gate[Branches×Dim] if filter).
 * x: [batch*seq*dim]; pre=x copy; post=[batch*seq*OutDim]. */
WELVET_API welvet_status welvet_parallel_forward_f32(
    int32_t batch, int32_t seq, int32_t dim, int32_t out_feat, int32_t branches, int32_t combine,
    int32_t sparse_k, int32_t dtype, int32_t backend, double disagree_beta, const float* packed,
    const float* x, float* pre, float* post);

/* Metal NormalBP; use_target=1 computes MSE; otherwise consumes gradOut.
 * dW/W_after use the packed forward weight layout; dX uses input layout. */
WELVET_API welvet_status welvet_parallel_normalbp_f32(
    int32_t batch, int32_t seq, int32_t dim, int32_t out_feat, int32_t branches, int32_t combine,
    int32_t sparse_k, int32_t dtype, int32_t backend, double disagree_beta, const float* packed,
    const float* x, const float* grad_or_target, int32_t use_target, double lr, float* pre,
    float* post, float* dX, float* dW, float* W_after, double* loss);

/* Metacognition — WMC1 wire. Observed Dense soft-double Dim×Dim FormatNone×34 + Metal rules.
 * packed: Observed [Dim×Dim]. DefaultStabilityRules when empty. */
WELVET_API welvet_status welvet_metacognition_forward_f32(
    int32_t batch, int32_t seq, int32_t dim, int32_t dtype, int32_t backend, const float* packed,
    const float* x, float* pre, float* post);

/* Metacognition NormalBP leaf. use_target: 0=gradOut on post, 1=MSE target. lr==0 skips SGD.
 * packed/dW/W_after: Observed [Dim×Dim]. */
WELVET_API welvet_status welvet_metacognition_normalbp_f32(
    int32_t batch, int32_t seq, int32_t dim, int32_t dtype, int32_t backend, const float* packed,
    const float* x, const float* grad_or_target, int32_t use_target, double lr, float* pre,
    float* post, float* dX, float* dW, float* W_after, double* loss);

#ifdef __cplusplus
}
#endif
