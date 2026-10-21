#pragma once

/* Stable C ABI for welvet.cpp — used by w2a go_v_cpp and future NAC. */

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

/* backend: 0 = CPU tiled (exact Go CPU path), 1 = imported SIMD DotTile path */
WELVET_API welvet_status welvet_dense_forward_f32(
    int32_t batch, int32_t in, int32_t out, int32_t act, int32_t dtype, int32_t backend,
    const float* W_init, const double* bias, size_t bias_len,
    const float* x, float* pre, float* post);

/* Back-compat wrappers (backend=CPU, dtype=Float32 for weights helper). */
WELVET_API welvet_status welvet_dense_forward_f32_cpu(
    int32_t batch, int32_t in, int32_t out, int32_t act, int32_t dtype,
    const float* W_init, const double* bias, size_t bias_len,
    const float* x, float* pre, float* post);

WELVET_API welvet_status welvet_dense_forward_f32_weights(
    int32_t batch, int32_t in, int32_t out, int32_t act,
    const float* W, const double* bias, size_t bias_len,
    const float* x, float* pre, float* post);

/* Dense NormalBP leaf: forward → (MSEGrad from target | use grad_out) → backward → optional SGD.
 * use_target: 0 = grad_or_target is gradOut; 1 = target (MSE).
 * lr == 0 skips ApplyGradSGD (W_after == W decoded).
 * Outputs: pre/post [batch*out], dX [batch*in], dW [out*in], W_after [out*in], *loss (0 if !use_target).
 */
WELVET_API welvet_status welvet_dense_normalbp_f32(
    int32_t batch, int32_t in, int32_t out, int32_t act, int32_t dtype, int32_t backend,
    const float* W_init, const double* bias, size_t bias_len, const float* x,
    const float* grad_or_target, int32_t use_target, double lr, float* pre, float* post, float* dX,
    float* dW, float* W_after, double* loss);

/* Architecture baseline: Depth×Rows×Cols×LayersPerCell + remote links.
 * links: n_links × 8 i32 = src z,y,x,l + tgt z,y,x,l
 * out_coords:  n_cells × 4 i32 (z,y,x,l) hop order
 * out_indices: n_cells × i32 Index(z,y,x,l)
 * out_remote:  n_cells × 5 i32 (is_remote, tz,ty,tx,tl)
 */
WELVET_API welvet_status welvet_arch_topology(
    int32_t depth, int32_t rows, int32_t cols, int32_t layers_per_cell,
    const int32_t* links, int32_t n_links, int32_t* out_n_cells, int32_t* out_coords,
    int32_t* out_indices, int32_t* out_remote);

/* MHA self-attn baseline (ModeSelf). mask/pos match Go mha.MaskKind/PosKind.
 * Q: [q_dim*d], K/V: [kv_dim*d], O: [d*q_dim], x: [batch*seq*d]
 * pre: [batch*seq*q_dim], post: [batch*seq*d]
 */
WELVET_API welvet_status welvet_mha_forward_f32(
    int32_t batch, int32_t seq, int32_t d_model, int32_t num_heads, int32_t num_kv_heads,
    int32_t head_dim, int32_t max_seq_len, int32_t mask, int32_t pos, int32_t dtype,
    int32_t backend, const float* Q, const float* K, const float* V, const float* O,
    const float* x, float* pre, float* post);

/* MHA NormalBP leaf (ModeSelf): forward → (MSEGrad | grad_out) → backward → optional SGD.
 * use_target: 0 = grad_or_target is gradOut [batch*seq*d]; 1 = target (MSE).
 * Outputs: pre [batch*seq*q_dim], post/dX [batch*seq*d], dW/W_after = concat(Q,K,V,O), *loss.
 */
WELVET_API welvet_status welvet_mha_normalbp_f32(
    int32_t batch, int32_t seq, int32_t d_model, int32_t num_heads, int32_t num_kv_heads,
    int32_t head_dim, int32_t max_seq_len, int32_t mask, int32_t pos, int32_t dtype,
    int32_t backend, const float* Q, const float* K, const float* V, const float* O,
    const float* x, const float* grad_or_target, int32_t use_target, double lr, float* pre,
    float* post, float* dX, float* dW, float* W_after, double* loss);

/* SwiGLU baseline. Gate/Up: [inter*in], Down: [in*inter], x: [batch*seq*in]
 * pre (h): [batch*seq*inter], post: [batch*seq*in]
 */
WELVET_API welvet_status welvet_swiglu_forward_f32(
    int32_t batch, int32_t seq, int32_t input_dim, int32_t intermediate_dim, int32_t dtype,
    int32_t backend, const float* Gate, const float* Up, const float* Down, const float* x,
    float* pre, float* post);

/* SwiGLU NormalBP leaf: forward → (MSEGrad from target | use grad_out) → backward → optional SGD.
 * use_target: 0 = grad_or_target is gradOut; 1 = target (MSE).
 * lr == 0 skips ApplyGradSGD.
 * Outputs: pre [batch*seq*inter], post/dX [batch*seq*in], dW/W_after = concat(Gate,Up,Down), *loss.
 */
WELVET_API welvet_status welvet_swiglu_normalbp_f32(
    int32_t batch, int32_t seq, int32_t input_dim, int32_t intermediate_dim, int32_t dtype,
    int32_t backend, const float* Gate, const float* Up, const float* Down, const float* x,
    const float* grad_or_target, int32_t use_target, double lr, float* pre, float* post, float* dX,
    float* dW, float* W_after, double* loss);

/* RMSNorm baseline. gamma: [dim], x: [batch*seq*dim]
 * pre (x̂): [batch*seq*dim], post (γ⊙x̂): [batch*seq*dim]
 * eps<=0 → 1e-6
 */
WELVET_API welvet_status welvet_rmsnorm_forward_f32(
    int32_t batch, int32_t seq, int32_t dim, double eps, int32_t dtype, int32_t backend,
    const float* gamma, const float* x, float* pre, float* post);

/* RMSNorm NormalBP leaf: forward → (MSEGrad from target | use grad_out) → backward → optional SGD.
 * use_target: 0 = grad_or_target is gradOut; 1 = target (MSE).
 * lr == 0 skips ApplyGradSGD (gamma_after == gamma decoded).
 * Outputs: pre/post/dX [batch*seq*dim], dW/gamma_after [dim], *loss (0 if !use_target).
 */
WELVET_API welvet_status welvet_rmsnorm_normalbp_f32(
    int32_t batch, int32_t seq, int32_t dim, double eps, int32_t dtype, int32_t backend,
    const float* gamma_init, const float* x, const float* grad_or_target, int32_t use_target,
    double lr, float* pre, float* post, float* dX, float* dW, float* gamma_after, double* loss);

/* LayerNorm baseline. gamma/beta: [dim], x: [batch*seq*dim]
 * pre (x̂): [batch*seq*dim], post (γ⊙x̂+β): [batch*seq*dim]
 * eps<=0 → 1e-5
 */
WELVET_API welvet_status welvet_layernorm_forward_f32(
    int32_t batch, int32_t seq, int32_t dim, double eps, int32_t dtype, int32_t backend,
    const float* gamma, const float* beta, const float* x, float* pre, float* post);

/* LayerNorm NormalBP leaf: forward → (MSEGrad from target | use grad_out) → backward → optional SGD.
 * use_target: 0 = grad_or_target is gradOut; 1 = target (MSE).
 * lr == 0 skips ApplyGradSGD.
 * Outputs: pre/post/dX [batch*seq*dim], dW [2*dim]=dγ|dβ, gamma_after/beta_after [dim], *loss.
 */
WELVET_API welvet_status welvet_layernorm_normalbp_f32(
    int32_t batch, int32_t seq, int32_t dim, double eps, int32_t dtype, int32_t backend,
    const float* gamma_init, const float* beta_init, const float* x, const float* grad_or_target,
    int32_t use_target, double lr, float* pre, float* post, float* dX, float* dW,
    float* gamma_after, float* beta_after, double* loss);

/* CNN1 Conv1d. W: [filters*inC*k], x: [batch*inC*seq]
 * pre/post loom: [batch*filters*outLen]
 */
WELVET_API welvet_status welvet_cnn1_forward_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t seq_len, int32_t kernel, int32_t stride,
    int32_t padding, int32_t act, int32_t dtype, int32_t backend, const float* W,
    const double* bias, size_t bias_len, const float* x, float* pre, float* post);

/* CNN1 NormalBP leaf: forward → (MSEGrad | grad_out) → backward → optional SGD.
 * use_target: 0 = grad_or_target is gradOut loom; 1 = target (MSE).
 * dW/W_after: [filters*inC*k]. lr==0 skips ApplyGradSGD.
 */
WELVET_API welvet_status welvet_cnn1_normalbp_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t seq_len, int32_t kernel, int32_t stride,
    int32_t padding, int32_t act, int32_t dtype, int32_t backend, const float* W,
    const double* bias, size_t bias_len, const float* x, const float* grad_or_target,
    int32_t use_target, double lr, float* pre, float* post, float* dX, float* dW, float* W_after,
    double* loss);

/* CNN2 Conv2d (square kernel). W: [filters*inC*k*k], x: [batch*inC*H*W]
 * pre/post loom: [batch*filters*outH*outW]
 */
WELVET_API welvet_status welvet_cnn2_forward_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t height, int32_t width, int32_t kernel,
    int32_t stride, int32_t padding, int32_t act, int32_t dtype, int32_t backend, const float* W,
    const double* bias, size_t bias_len, const float* x, float* pre, float* post);

WELVET_API welvet_status welvet_cnn2_normalbp_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t height, int32_t width, int32_t kernel,
    int32_t stride, int32_t padding, int32_t act, int32_t dtype, int32_t backend, const float* W,
    const double* bias, size_t bias_len, const float* x, const float* grad_or_target,
    int32_t use_target, double lr, float* pre, float* post, float* dX, float* dW, float* W_after,
    double* loss);

/* CNN3 Conv3d (cubic kernel). W: [filters*inC*k³], x: [batch*inC*D*H*W]
 * pre/post loom: [batch*filters*outD*outH*outW]
 */
WELVET_API welvet_status welvet_cnn3_forward_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t depth, int32_t height, int32_t width,
    int32_t kernel, int32_t stride, int32_t padding, int32_t act, int32_t dtype, int32_t backend,
    const float* W, const double* bias, size_t bias_len, const float* x, float* pre, float* post);

WELVET_API welvet_status welvet_cnn3_normalbp_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t depth, int32_t height, int32_t width,
    int32_t kernel, int32_t stride, int32_t padding, int32_t act, int32_t dtype, int32_t backend,
    const float* W, const double* bias, size_t bias_len, const float* x, const float* grad_or_target,
    int32_t use_target, double lr, float* pre, float* post, float* dX, float* dW, float* W_after,
    double* loss);

/* ConvT1/2/3 — transposed conv (scatter). No bias. output_padding in OutLen only. */
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

/* ConvT NormalBP leaf (no bias). use_target: 0=gradOut, 1=MSE target. lr==0 skips SGD. */
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

/* RNN. packed: [IH (H×In) | HH (H×H) | bias (H)], x: [batch*seq*in]
 * pre/post: [batch*seq*H]
 */
WELVET_API welvet_status welvet_rnn_forward_f32(
    int32_t batch, int32_t input_size, int32_t hidden_size, int32_t seq_len, int32_t dtype,
    int32_t backend, const float* packed, const float* x, float* pre, float* post);

/* LSTM. packed: [i|f|g|o] each [IH|HH|bias], x: [batch*seq*in]
 * pre: [batch*seq*5H]=[i,f,g,o,c], post: [batch*seq*H]
 */
WELVET_API welvet_status welvet_lstm_forward_f32(
    int32_t batch, int32_t input_size, int32_t hidden_size, int32_t seq_len, int32_t dtype,
    int32_t backend, const float* packed, const float* x, float* pre, float* post);

/* RNN/LSTM NormalBP leaf. use_target: 0=gradOut on post, 1=MSE target. lr==0 skips SGD.
 * dW/W_after: loom pack (RNN [IH|HH|bias]; LSTM 4×gate).
 */
WELVET_API welvet_status welvet_rnn_normalbp_f32(
    int32_t batch, int32_t input_size, int32_t hidden_size, int32_t seq_len, int32_t dtype,
    int32_t backend, const float* packed, const float* x, const float* grad_or_target,
    int32_t use_target, double lr, float* pre, float* post, float* dX, float* dW, float* W_after,
    double* loss);

WELVET_API welvet_status welvet_lstm_normalbp_f32(
    int32_t batch, int32_t input_size, int32_t hidden_size, int32_t seq_len, int32_t dtype,
    int32_t backend, const float* packed, const float* x, const float* grad_or_target,
    int32_t use_target, double lr, float* pre, float* post, float* dX, float* dW, float* W_after,
    double* loss);

/* Embedding gather. W: [vocab*emb], x: [batch*seq*in_last] token ids as f32
 * in_rank 1..3; in_last 1 (or emb for chain). pre/post: [batch*seq*emb] (rank1→seq*emb)
 */
WELVET_API welvet_status welvet_embedding_forward_f32(
    int32_t batch, int32_t vocab, int32_t emb, int32_t seq_len, int32_t in_rank, int32_t in_last,
    int32_t act, int32_t dtype, int32_t backend, const float* W, const float* x, float* pre,
    float* post);

/* Embedding NormalBP leaf: gather → (MSEGrad | grad_out) → scatter dW → optional SGD.
 * use_target: 0 = grad_or_target is gradOut [out_n]; 1 = target (MSE) same layout as post.
 * Outputs: pre/post [out_n], dX zeros [x_n], dW/W_after [vocab*emb], *loss.
 */
WELVET_API welvet_status welvet_embedding_normalbp_f32(
    int32_t batch, int32_t vocab, int32_t emb, int32_t seq_len, int32_t in_rank, int32_t in_last,
    int32_t act, int32_t dtype, int32_t backend, const float* W_init, const float* x,
    const float* grad_or_target, int32_t use_target, double lr, float* pre, float* post, float* dX,
    float* dW, float* W_after, double* loss);

/* Softmax (weightless). All Kind variants. x: [n] flat or shaped via rank/dims.
 * kind: 0 std 1 temp 2 gumbel 3 masked 4 sparse 5 entmax 6 grid 7 hierarchical
 * mask: uint8 0/1 length mask_len (KindMasked); hier: int32 levels (KindHierarchical)
 * pre/post: probabilities, length n
 */
WELVET_API welvet_status welvet_softmax_forward_f32(
    int32_t n, int32_t dim, int32_t seq_len, int32_t kind, int32_t rows, int32_t cols,
    double temperature, double entmax_alpha, int32_t gumbel_seed, int32_t dtype, int32_t backend,
    int32_t rank, const int32_t* shape, const uint8_t* mask, int32_t mask_len, const int32_t* hier,
    int32_t hier_len, const float* x, float* pre, float* post);

/* Softmax NormalBP leaf: forward → (MSEGrad | grad_out) → Jacobian dX; weightless.
 * use_target: 0 = grad_or_target is gradOut; 1 = target (MSE). lr: ApplyGradSGD no-op.
 * Outputs: pre/post/dX [n], *loss (0 if !use_target).
 */
WELVET_API welvet_status welvet_softmax_normalbp_f32(
    int32_t n, int32_t dim, int32_t seq_len, int32_t kind, int32_t rows, int32_t cols,
    double temperature, double entmax_alpha, int32_t gumbel_seed, int32_t dtype, int32_t backend,
    int32_t rank, const int32_t* shape, const uint8_t* mask, int32_t mask_len, const int32_t* hier,
    int32_t hier_len, const float* x, const float* grad_or_target, int32_t use_target, double lr,
    float* pre, float* post, float* dX, double* loss);

/* Soft KMeans. W: [K*D] centers, x: flat input last-dim=D.
 * output_mode: 0=probabilities (out K), 1=features (out D). pre/post length batch*out_dim
 */
WELVET_API welvet_status welvet_kmeans_forward_f32(
    int32_t n, int32_t feature_dim, int32_t num_clusters, int32_t output_mode, int32_t act,
    double temperature, int32_t dtype, int32_t backend, int32_t rank, const int32_t* shape,
    const float* W, const float* x, float* pre, float* post);

/* Soft KMeans NormalBP leaf. W: [K*D] centers; x: [batch*seq*D] (seq=1 → [batch*D]).
 * output_mode: 0=probabilities (out K), 1=features (out D).
 * use_target: 0=gradOut on post, 1=MSE target. lr==0 skips SGD.
 * pre/post/dX: batch*seq*out_dim; dW/W_after: [K*D].
 */
WELVET_API welvet_status welvet_kmeans_normalbp_f32(
    int32_t batch, int32_t seq, int32_t num_clusters, int32_t feature_dim, int32_t output_mode,
    int32_t act, double temperature, int32_t dtype, int32_t backend, const float* W, const float* x,
    const float* grad_or_target, int32_t use_target, double lr, float* pre, float* post, float* dX,
    float* dW, float* W_after, double* loss);

/* Sequential Dense chain. packed: Depth×[Dim×Dim], x: [batch*seq*dim] (seq=1 → [batch*dim])
 * pre/post: same layout as x (last child). child_act typically Linear (-1).
 */
WELVET_API welvet_status welvet_sequential_forward_f32(
    int32_t batch, int32_t seq, int32_t dim, int32_t depth, int32_t child_act, int32_t dtype,
    int32_t backend, const float* packed, const float* x, float* pre, float* post);

/* Sequential NormalBP leaf. use_target: 0=gradOut on post, 1=MSE target. lr==0 skips SGD.
 * dW/W_after: loom Depth×[Dim×Dim].
 */
WELVET_API welvet_status welvet_sequential_normalbp_f32(
    int32_t batch, int32_t seq, int32_t dim, int32_t depth, int32_t child_act, int32_t dtype,
    int32_t backend, const float* packed, const float* x, const float* grad_or_target,
    int32_t use_target, double lr, float* pre, float* post, float* dX, float* dW, float* W_after,
    double* loss);

/* Residual y=F(x)+x. packed: Depth×[Dim×Dim] Dense F, x: [batch*seq*dim] (seq=1 → [batch*dim])
 * pre: last F child pre; post: skip-add result. child_act typically Linear (-1).
 */
WELVET_API welvet_status welvet_residual_forward_f32(
    int32_t batch, int32_t seq, int32_t dim, int32_t depth, int32_t child_act, int32_t dtype,
    int32_t backend, const float* packed, const float* x, float* pre, float* post);

/* Residual NormalBP leaf. use_target: 0=gradOut on post, 1=MSE target. lr==0 skips SGD.
 * dW/W_after: loom Depth×[Dim×Dim] (F children only).
 */
WELVET_API welvet_status welvet_residual_normalbp_f32(
    int32_t batch, int32_t seq, int32_t dim, int32_t depth, int32_t child_act, int32_t dtype,
    int32_t backend, const float* packed, const float* x, const float* grad_or_target,
    int32_t use_target, double lr, float* pre, float* post, float* dX, float* dW, float* W_after,
    double* loss);

/* Mamba minimal SSM. packed: InProj[2*Inner×D]|OutProj[D×Inner]|ALog[Inner]|DSkip[Inner]
 * x: [batch*seq*d_model]; pre/post: same layout. Inner = expand*d_model (expand≤0 → 2).
 */
WELVET_API welvet_status welvet_mamba_forward_f32(
    int32_t batch, int32_t seq, int32_t d_model, int32_t d_state, int32_t expand, int32_t dtype,
    int32_t backend, const float* packed, const float* x, float* pre, float* post);

/* Mamba NormalBP leaf. use_target: 0=gradOut on post, 1=MSE target. lr==0 skips SGD.
 * packed/dW/W_after: InProj|OutProj|ALog|DSkip.
 */
WELVET_API welvet_status welvet_mamba_normalbp_f32(
    int32_t batch, int32_t seq, int32_t d_model, int32_t d_state, int32_t expand, int32_t dtype,
    int32_t backend, const float* packed, const float* x, const float* grad_or_target,
    int32_t use_target, double lr, float* pre, float* post, float* dX, float* dW, float* W_after,
    double* loss);

/* Metacognition Observed Dense Dim→Dim + default stability rules (first fwd = Dense while cool).
 * packed: [Dim×Dim]; x: [batch*seq*dim] (seq=1 → [batch*dim]); pre/post same layout.
 */
WELVET_API welvet_status welvet_metacognition_forward_f32(
    int32_t batch, int32_t seq, int32_t dim, int32_t dtype, int32_t backend, const float* packed,
    const float* x, float* pre, float* post);

/* Metacognition NormalBP leaf. use_target: 0=gradOut on post, 1=MSE target. lr==0 skips SGD.
 * packed/dW/W_after: Observed [Dim×Dim].
 */
WELVET_API welvet_status welvet_metacognition_normalbp_f32(
    int32_t batch, int32_t seq, int32_t dim, int32_t dtype, int32_t backend, const float* packed,
    const float* x, const float* grad_or_target, int32_t use_target, double lr, float* pre,
    float* post, float* dX, float* dW, float* W_after, double* loss);

/* GDN (Gated DeltaNet). Float32 FormatNone. packed: InQKV|InZ|InB|InA|Out|Conv|ALog|DtBias|γ
 * x/pre/post: [batch*seq*hidden]. dtype ignored (must be Float32).
 */
WELVET_API welvet_status welvet_gdn_forward_f32(
    int32_t batch, int32_t seq, int32_t hidden, int32_t num_key_heads, int32_t num_value_heads,
    int32_t key_head_dim, int32_t value_head_dim, int32_t conv_kernel, int32_t dtype, int32_t backend,
    const float* packed, const float* x, float* pre, float* post);

/* GDN truncated-BPTT NormalBP leaf (f32). use_target: 0=gradOut on post, 1=MSE target. lr==0 skips SGD.
 * packed/dW/W_after: InQKV|InZ|InB|InA|Out|Conv|ALog|DtBias|γ. dtype ignored or must be Float32.
 */
WELVET_API welvet_status welvet_gdn_normalbp_f32(
    int32_t batch, int32_t seq, int32_t hidden, int32_t num_key_heads, int32_t num_value_heads,
    int32_t key_head_dim, int32_t value_head_dim, int32_t conv_kernel, int32_t dtype, int32_t backend,
    const float* packed, const float* x, const float* grad_or_target, int32_t use_target, double lr,
    float* pre, float* post, float* dX, float* dW, float* W_after, double* loss);

/* Stack NestedStack of Dense Dim→Dim NormalBP leaf. packed: n_children×[Dim×Dim], x: [batch*seq*dim]
 * (seq=1 → [batch*dim]). use_target: 0=gradOut on post, 1=MSE target. lr==0 skips SGD.
 * dW/W_after: loom n_children×[Dim×Dim]. Prefer STKB/STKO (not STB1 StepBP).
 */
WELVET_API welvet_status welvet_stack_normalbp_f32(
    int32_t batch, int32_t seq, int32_t dim, int32_t n_children, int32_t child_act, int32_t dtype,
    int32_t backend, const float* packed, const float* x, const float* grad_or_target,
    int32_t use_target, double lr, float* pre, float* post, float* dX, float* dW, float* W_after,
    double* loss);

/* seqmix Kind name. kind: 0=attention 1=ssm 2=linear_attn 3=conv_mix.
 * Writes NUL-terminated name into buf (max buf_len). Returns Ok or InvalidArgument.
 */
WELVET_API welvet_status welvet_seqmix_kind_name(int32_t kind, char* buf, int32_t buf_len);

/* Mixer Kind tags: mha→0, mamba→1, gdn→2 (matches seqmix.Kind iota). */
WELVET_API int32_t welvet_seqmix_mha_kind(void);
WELVET_API int32_t welvet_seqmix_mamba_kind(void);
WELVET_API int32_t welvet_seqmix_gdn_kind(void);

/* Parallel Dense branches. combine: 0 concat,1 add,2 avg,3 filter,4 max,5 sparsek,6 disagree.
 * packed: Branches×[OutFeat×Dim] (+ Gate[Branches×Dim] if filter).
 * x: [batch*seq*dim]; pre=x copy; post=[batch*seq*OutDim].
 */
WELVET_API welvet_status welvet_parallel_forward_f32(
    int32_t batch, int32_t seq, int32_t dim, int32_t out_feat, int32_t branches, int32_t combine,
    int32_t sparse_k, int32_t dtype, int32_t backend, double disagree_beta, const float* packed,
    const float* x, float* pre, float* post);

/* Parallel Dense NormalBP leaf. Same packing as welvet_parallel_forward_f32.
 * use_target: 0=gradOut on post, 1=MSE target. lr==0 skips SGD.
 * dW/W_after: same packed layout as input. Prefer WPLB/WPLO for multi-kind.
 */
WELVET_API welvet_status welvet_parallel_normalbp_f32(
    int32_t batch, int32_t seq, int32_t dim, int32_t out_feat, int32_t branches, int32_t combine,
    int32_t sparse_k, int32_t dtype, int32_t backend, double disagree_beta, const float* packed,
    const float* x, const float* grad_or_target, int32_t use_target, double lr, float* pre,
    float* post, float* dX, float* dW, float* W_after, double* loss);

/* Line StepBP (FamilyBP on pop) over a thin Stack of Dense children.
 * packed_W / W_after: concat of child [out×in] matrices. Bias omitted (empty).
 * x_ticks: n_ticks × [batch × in_h[0]]; target_ticks: n_ticks × [batch × out_h[last]].
 * loss: last RoleTrain pop MSE (0 if pipe not yet flushed). Prefer STB1/STBO for multi-kind.
 */
WELVET_API welvet_status welvet_stack_stepbp_dense_f32(
    int32_t n_children, int32_t n_ticks, int32_t batch, double lr, int32_t dtype,
    const int32_t* in_h, const int32_t* out_h, const int32_t* act, const float* packed_W,
    const float* x_ticks, const float* target_ticks, float* W_after, double* loss);

/* Stack TrainMSE dispatcher by TrainMode (Go parallel.TrainMode iota).
 * Packing matches welvet_stack_normalbp_f32 (NestedStack Dense Dim→Dim).
 * target: [batch*seq*dim]. Step/Mesh modes use same family credit as non-Step leaf.
 * Shadow==Freeze; Adversarial/Memory==FamilyBP. Exports W_after + loss.
 */
WELVET_API welvet_status welvet_stack_train_mse(
    int32_t mode, int32_t batch, int32_t seq, int32_t dim, int32_t n_children, int32_t child_act,
    int32_t dtype, int32_t backend, const float* packed, const float* x, const float* target,
    double lr, float* W_after, double* loss);

/* Parallel TrainMSE dispatcher by TrainMode. Packing matches welvet_parallel_forward_f32 /
 * welvet_parallel_normalbp_f32. target: [batch*seq*OutDim]. Exports W_after + loss.
 */
WELVET_API welvet_status welvet_parallel_train_mse(
    int32_t mode, int32_t batch, int32_t seq, int32_t dim, int32_t out_feat, int32_t branches,
    int32_t combine, int32_t sparse_k, int32_t dtype, int32_t backend, double disagree_beta,
    const float* packed, const float* x, const float* target, double lr, float* W_after,
    double* loss);

/* Mesh-clock TrainMSE (runtime/step) by TrainMode.
 * Places the same OpKind in every hop-order cell:
 *   cell_kind 1  = Dense Dim→Dim (n_sub ignored)
 *   cell_kind 19 = NestedStack of n_sub Dense Dim→Dim
 *   cell_kind 23 = Parallel (n_sub branches, Combine=Add, out_feat=dim)
 * packed: concat per cell of that cell's pack (Dense/Stack/Parallel layout).
 * x/target: [batch*dim]. Mesh* → step Credit; non-Mesh → FamilyBP.
 */
WELVET_API welvet_status welvet_grid_train_mse(
    int32_t mode, int32_t depth, int32_t rows, int32_t cols, int32_t lpc, int32_t batch,
    int32_t ticks, int32_t dim, int32_t cell_kind, int32_t n_sub, int32_t child_act, int32_t dtype,
    int32_t backend, const float* packed, const float* x, const float* target, double lr,
    float* W_after, double* loss);

/* Line (systolic) TrainMSE by TrainMode over a thin Stack of Dense children.
 * Packing matches welvet_stack_stepbp_dense_f32. Step* modes map to line Credit;
 * non-Step → FamilyBP. n_ticks RoleTrain samples; loss = last pop MSE.
 */
WELVET_API welvet_status welvet_stack_line_train_mse(
    int32_t mode, int32_t n_children, int32_t n_ticks, int32_t batch, double lr, int32_t dtype,
    const int32_t* in_h, const int32_t* out_h, const int32_t* act, const float* packed_W,
    const float* x_ticks, const float* target_ticks, float* W_after, double* loss);

WELVET_API const char* welvet_version(void);
WELVET_API int32_t welvet_gap_count(void);
WELVET_API void welvet_print_gaps(void);

#ifdef __cplusplus
}
#endif
