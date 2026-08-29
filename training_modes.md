# Welvet training modes

Legend: `[T]=Tween` · `[S]=Split` · `[FP]=FastProxy` · `[L]=Linear` · `[HP]=HeadProxy`

All **29** named credit modes used by tide / test53 `AllModes`. First six keep Lucy checkpoint aliases (`sgd`, `step_sgd`, …); the rest use the Welvet `String()` name as the token. Dashboards print `ShortTrainMode`; persistence still uses the full / CLI token.

## How credit works (shared)

Sandwich update via `TrainStackMSE` / `TrainStackCE` / `OpenSplitTape`.

- **Output gap** \(g_y\): regression uses MSE residual; classification uses softmax − one-hot (CE).
- **\(P(\cdot)\)**: project / reshape the gap onto a leaf’s activation shape.
- **\(N\)**: number of trainable leaves on the sandwich (layers / cam branches).
- **Step\***: 1D systolic pipe (`TrainLine`) — one sample enters layer 0 per tick; fill ticks do not update. Same credit family as the non-Step twin on Stack.
- **Mesh\***: needs volumetric / grid placement; on a plain Stack it often collapses to the family update.
- Rivaling backprop = matched **hard Acc** vs `StepBP`, not Lucy Score. Sparse wins Score via Availability (skip-GEMV). FastProxy is the Acc rival on sine/copy toys.

## Mode table

| # | Full name (Welvet) | Short | Ckpt / CLI token | Equation / how it works |
|---|--------------------|-------|------------------|-------------------------|
| 1 | NormalBP | NormalBP | `sgd` | Real chain rule: `BackwardStack` + SGD. \(\delta_\ell = (W_{\ell+1}^\top \delta_{\ell+1}) \odot \sigma'(z_\ell)\), \(dW_\ell = \delta_\ell x_\ell^\top\). |
| 2 | StepBP | StepBP | `step_sgd` | Same chain rule as NormalBP, but on the **Step** pipe (one layer’s worth of update per tick; fill ticks skip). |
| 3 | Tween | [T] | `tween` | Broadcast the output gap to every leaf: \(g_\ell = P(g_y)\), typically **half LR**. No chain rule. |
| 4 | TweenChain | [T]Chain | `tween_chain` | Chain rule on Stack (= BP credit), Tween naming / scheduling. |
| 5 | StepTween | Step[T] | `step_tween` | Tween broadcast on the Step pipe. |
| 6 | StepTweenChain | Step[T]Chain | `step_tween_chain` | TweenChain (BP) on the Step pipe. |
| 7 | MeshBP | MeshBP | `MeshBP` | Chain-rule BP with **Mesh** / volumetric scheduling. |
| 8 | MeshTween | Mesh[T] | `MeshTween` | Tween broadcast under Mesh scheduling. |
| 9 | MeshTweenChain | Mesh[T]Chain | `MeshTweenChain` | Chain-rule (= BP) under Mesh scheduling. |
| 10 | TweenSplit | [T][S] | `TweenSplit` | One global gap, split evenly across leaves: \(g_i = \frac{1}{N} P(g_y)\). No chain rule. |
| 11 | StepTweenSplit | Step[T][S] | `StepTweenSplit` | TweenSplit credit on the Step pipe. |
| 12 | TweenAlt | [T]Alt | `TweenAlt` | Alternates Split then Tween (`AltTimes` cycles). |
| 13 | StepTweenAlt | Step[T]Alt | `StepTweenAlt` | TweenAlt on the Step pipe. |
| 14 | TweenSplitHeadProxy | [T][S][HP] | `TweenSplitHeadProxy` | Head gets true \(J^\top g_y\); hidden leaves get **\(dW\) only** (no full \(\delta\) walk). |
| 15 | TweenSplitLinear | [T][S][L] | `TweenSplitLinear` | Affine \(W^\top\) walk of the gap; **skips** activation derivative \(\sigma'\). |
| 16 | TweenSplitFastProxy | [T][S][FP] | `TweenSplitFastProxy` | \(g_{\mathrm{proxy}} = W_{\mathrm{head}}^\top g_y\) (no \(\sigma'\)); hidden leaves **\(dW\) only**. |
| 17 | TweenSplitLinearCache | [T][S][L]Cache | `TweenSplitLinearCache` | Cached Linear walk (can go dead on sine freq switch; kept for A/B). |
| 18 | TweenSplitHeadProxyAsync | [T][S][HP]Async | `TweenSplitHeadProxyAsync` | Like HeadProxy, but hidden uses the proxy from sample \(T-1\) (async). |
| 19 | TweenSplitSparse | [T][S]Sparse | `TweenSplitSparse` | Head always updates; **one rotating** hidden leaf per step (skip-GEMV → high Avail). |
| 20 | MeshTweenSplit | Mesh[T][S] | `MeshTweenSplit` | TweenSplit under Mesh / grid schedule (family collapse on plain Stack). |
| 21 | MeshTweenAlt | Mesh[T]Alt | `MeshTweenAlt` | TweenAlt under Mesh schedule. |
| 22 | MeshTweenSplitFastProxy | Mesh[T][S][FP] | `MeshTweenSplitFastProxy` | FastProxy credit under Mesh schedule. |
| 23 | MeshTweenSplitSparse | Mesh[T][S]Sparse | `MeshTweenSplitSparse` | Sparse credit under Mesh schedule. |
| 24 | StepTweenSplitHeadProxy | Step[T][S][HP] | `StepTweenSplitHeadProxy` | HeadProxy on the Step pipe. |
| 25 | StepTweenSplitLinear | Step[T][S][L] | `StepTweenSplitLinear` | Linear (no \(\sigma'\)) on the Step pipe. |
| 26 | StepTweenSplitFastProxy | Step[T][S][FP] | `StepTweenSplitFastProxy` | FastProxy on the Step pipe. |
| 27 | StepTweenSplitLinearCache | Step[T][S][L]Cache | `StepTweenSplitLinearCache` | LinearCache on the Step pipe. |
| 28 | StepTweenSplitHeadProxyAsync | Step[T][S][HP]Async | `StepTweenSplitHeadProxyAsync` | HeadProxyAsync on the Step pipe. |
| 29 | StepTweenSplitSparse | Step[T][S]Sparse | `StepTweenSplitSparse` | Sparse on the Step pipe. |

## Families at a glance

| Family | Core idea |
|--------|-----------|
| **BP** (`NormalBP` / `StepBP` / `MeshBP`) | True backprop chain rule |
| **Tween** | Broadcast \(P(g_y)\) (often half LR) |
| **TweenChain** | Named Tween, credit = BP |
| **TweenSplit** | \(g_i = \frac1N P(g_y)\) per leaf |
| **TweenAlt** | Time-multiplex Split ↔ Tween |
| **HeadProxy** | True head Jacobian; hidden \(dW\) only |
| **Linear** | \(W^\top\) walk, skip \(\sigma'\) |
| **FastProxy** | \(g_{\mathrm{proxy}} = W_{\mathrm{head}}^\top g_y\); hidden \(dW\) only |
| **HeadProxyAsync** | Hidden proxy delayed by one sample |
| **Sparse** | Head + one rotating hidden leaf |
| **Step\*** prefix | Same credit, systolic 1D pipe |
| **Mesh\*** prefix | Same credit, volumetric / grid schedule |

## Cameral note

Cameral nets use `BranchModes` / `HemispheresFrom` / `CombineAdd`: \(n=1\) one mid Op; \(n=2/3\) Bi/Tri. Split-family credit still divides by leaf count \(N\) across branches.

See also README § “Training credit — scorecard §9”.
