package parallel

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/convt1"
	"github.com/openfluke/welvet/layers/convt2"
	"github.com/openfluke/welvet/layers/convt3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mamba"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
)

// branchForward dispatches one branch Op. Dense branches receive flattened [n,dim]
// when the Parallel input was sequence-shaped; all other Ops get the original input.
func branchForward[T core.Numeric](op any, input, flat *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if op == nil {
		return nil, nil, fmt.Errorf("parallel: nil branch")
	}
	switch v := op.(type) {
	case *dense.Layer:
		in := input
		if flat != nil {
			in = flat
		}
		return dense.Forward(v, in)
	case *mha.Layer:
		return mha.Forward(v, input)
	case *swiglu.Layer:
		return swiglu.Forward(v, input)
	case *rmsnorm.Layer:
		return rmsnorm.Forward(v, input)
	case *layernorm.Layer:
		return layernorm.Forward(v, input)
	case *softmax.Layer:
		return softmax.Forward(v, input)
	case *cnn1.Layer:
		return cnn1.Forward(v, input)
	case *cnn2.Layer:
		return cnn2.Forward(v, input)
	case *cnn3.Layer:
		return cnn3.Forward(v, input)
	case *convt1.Layer:
		return convt1.Forward(v, input)
	case *convt2.Layer:
		return convt2.Forward(v, input)
	case *convt3.Layer:
		return convt3.Forward(v, input)
	case *rnn.Layer:
		return rnn.Forward(v, input)
	case *lstm.Layer:
		return lstm.Forward(v, input)
	case *embedding.Layer:
		return embedding.Forward(v, input)
	case *sequential.Layer:
		return sequential.Forward(v, input)
	case *residual.Layer:
		return residual.Forward(v, input)
	case *Layer:
		return Forward(v, input)
	case *kmeans.Layer:
		return kmeans.Forward(v, input)
	case *mamba.Layer:
		return mamba.Forward(v, input)
	case *metacognition.Layer:
		return metacognition.Forward(v, input)
	case *gdn.Layer:
		return gdn.Forward(v, input)
	default:
		return nil, nil, fmt.Errorf("parallel: unsupported branch Op %T (no silent Dense fallback)", op)
	}
}

func branchBackward[T core.Numeric](op any, gradOut, input, flat, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if op == nil {
		return nil, nil, fmt.Errorf("parallel: nil branch")
	}
	switch v := op.(type) {
	case *dense.Layer:
		in := input
		if flat != nil {
			in = flat
		}
		return dense.Backward(v, gradOut, in, pre)
	case *mha.Layer:
		return mha.Backward(v, gradOut, input, pre)
	case *swiglu.Layer:
		return swiglu.Backward(v, gradOut, input, pre)
	case *rmsnorm.Layer:
		return rmsnorm.Backward(v, gradOut, input, pre)
	case *layernorm.Layer:
		return layernorm.Backward(v, gradOut, input, pre)
	case *softmax.Layer:
		return softmax.Backward(v, gradOut, input, pre)
	case *cnn1.Layer:
		return cnn1.Backward(v, gradOut, input, pre)
	case *cnn2.Layer:
		return cnn2.Backward(v, gradOut, input, pre)
	case *cnn3.Layer:
		return cnn3.Backward(v, gradOut, input, pre)
	case *convt1.Layer:
		return convt1.Backward(v, gradOut, input, pre)
	case *convt2.Layer:
		return convt2.Backward(v, gradOut, input, pre)
	case *convt3.Layer:
		return convt3.Backward(v, gradOut, input, pre)
	case *rnn.Layer:
		return rnn.Backward(v, gradOut, input, pre)
	case *lstm.Layer:
		return lstm.Backward(v, gradOut, input, pre)
	case *embedding.Layer:
		return embedding.Backward(v, gradOut, input, pre)
	case *sequential.Layer:
		return sequential.Backward(v, gradOut, input, pre)
	case *residual.Layer:
		return residual.Backward(v, gradOut, input, pre)
	case *Layer:
		return Backward(v, gradOut, input, pre)
	case *kmeans.Layer:
		return kmeans.Backward(v, gradOut, input, pre)
	case *mamba.Layer:
		return mamba.Backward(v, gradOut, input, pre)
	case *metacognition.Layer:
		return metacognition.Backward(v, gradOut, input, pre)
	case *gdn.Layer:
		return gdn.Backward(v, gradOut, input, pre)
	default:
		return nil, nil, fmt.Errorf("parallel: unsupported branch Op %T (no silent Dense fallback)", op)
	}
}

func branchPack(op any, format quant.Format) error {
	switch v := op.(type) {
	case *dense.Layer:
		return v.Pack(format)
	case *mha.Layer:
		return v.Pack(format)
	case *swiglu.Layer:
		return v.Pack(format)
	case *rmsnorm.Layer:
		return v.Pack(format)
	case *layernorm.Layer:
		return v.Pack(format)
	case *softmax.Layer:
		return v.Pack(format)
	case *cnn1.Layer:
		return v.Pack(format)
	case *cnn2.Layer:
		return v.Pack(format)
	case *cnn3.Layer:
		return v.Pack(format)
	case *convt1.Layer:
		return v.Pack(format)
	case *convt2.Layer:
		return v.Pack(format)
	case *convt3.Layer:
		return v.Pack(format)
	case *rnn.Layer:
		return v.Pack(format)
	case *lstm.Layer:
		return v.Pack(format)
	case *embedding.Layer:
		return v.Pack(format)
	case *sequential.Layer:
		return v.Pack(format)
	case *residual.Layer:
		return v.Pack(format)
	case *Layer:
		return v.Pack(format)
	case *kmeans.Layer:
		return v.Pack(format)
	case *mamba.Layer:
		return v.Pack(format)
	case *metacognition.Layer:
		return v.Pack(format)
	case *gdn.Layer:
		return v.Pack(format)
	default:
		return fmt.Errorf("parallel: Pack unsupported branch %T", op)
	}
}

func branchSetDType(op any, dt core.DType) error {
	switch v := op.(type) {
	case *dense.Layer:
		return v.SetDType(dt)
	case *mha.Layer:
		return v.SetDType(dt)
	case *swiglu.Layer:
		return v.SetDType(dt)
	case *rmsnorm.Layer:
		return v.SetDType(dt)
	case *layernorm.Layer:
		return v.SetDType(dt)
	case *softmax.Layer:
		return v.SetDType(dt)
	case *cnn1.Layer:
		return v.SetDType(dt)
	case *cnn2.Layer:
		return v.SetDType(dt)
	case *cnn3.Layer:
		return v.SetDType(dt)
	case *convt1.Layer:
		return v.SetDType(dt)
	case *convt2.Layer:
		return v.SetDType(dt)
	case *convt3.Layer:
		return v.SetDType(dt)
	case *rnn.Layer:
		return v.SetDType(dt)
	case *lstm.Layer:
		return v.SetDType(dt)
	case *embedding.Layer:
		return v.SetDType(dt)
	case *sequential.Layer:
		return v.SetDType(dt)
	case *residual.Layer:
		return v.SetDType(dt)
	case *Layer:
		return v.SetDType(dt)
	case *kmeans.Layer:
		return v.SetDType(dt)
	case *mamba.Layer:
		return v.SetDType(dt)
	case *metacognition.Layer:
		return v.SetDType(dt)
	case *gdn.Layer:
		// GDN projections are quant.Blob — no Store dtype axis.
		return nil
	default:
		return fmt.Errorf("parallel: SetDType unsupported branch %T", op)
	}
}

func branchGradWSize(op any) int {
	switch v := op.(type) {
	case *dense.Layer:
		return v.GradWSize()
	case *mha.Layer:
		return v.GradWSize()
	case *swiglu.Layer:
		return v.GradWSize()
	case *rmsnorm.Layer:
		return v.GradWSize()
	case *layernorm.Layer:
		return v.GradWSize()
	case *softmax.Layer:
		return v.GradWSize()
	case *cnn1.Layer:
		return v.GradWSize()
	case *cnn2.Layer:
		return v.GradWSize()
	case *cnn3.Layer:
		return v.GradWSize()
	case *convt1.Layer:
		return v.GradWSize()
	case *convt2.Layer:
		return v.GradWSize()
	case *convt3.Layer:
		return v.GradWSize()
	case *rnn.Layer:
		return v.GradWSize()
	case *lstm.Layer:
		return v.GradWSize()
	case *embedding.Layer:
		return v.GradWSize()
	case *sequential.Layer:
		return v.GradWSize()
	case *residual.Layer:
		return v.GradWSize()
	case *Layer:
		return v.GradWSize()
	case *kmeans.Layer:
		return v.GradWSize()
	case *mamba.Layer:
		return v.GradWSize()
	case *metacognition.Layer:
		return v.GradWSize()
	case *gdn.Layer:
		return v.GradWSize()
	default:
		return 0
	}
}

func branchApplyGradSGD[T core.Numeric](op any, dW *core.Tensor[T], lr float64) error {
	switch v := op.(type) {
	case *dense.Layer:
		return dense.ApplyGradSGD(v, dW, lr)
	case *mha.Layer:
		return mha.ApplyGradSGD(v, dW, lr)
	case *swiglu.Layer:
		return swiglu.ApplyGradSGD(v, dW, lr)
	case *rmsnorm.Layer:
		return rmsnorm.ApplyGradSGD(v, dW, lr)
	case *layernorm.Layer:
		return layernorm.ApplyGradSGD(v, dW, lr)
	case *softmax.Layer:
		return softmax.ApplyGradSGD(v, dW, lr)
	case *cnn1.Layer:
		return cnn1.ApplyGradSGD(v, dW, lr)
	case *cnn2.Layer:
		return cnn2.ApplyGradSGD(v, dW, lr)
	case *cnn3.Layer:
		return cnn3.ApplyGradSGD(v, dW, lr)
	case *convt1.Layer:
		return convt1.ApplyGradSGD(v, dW, lr)
	case *convt2.Layer:
		return convt2.ApplyGradSGD(v, dW, lr)
	case *convt3.Layer:
		return convt3.ApplyGradSGD(v, dW, lr)
	case *rnn.Layer:
		return rnn.ApplyGradSGD(v, dW, lr)
	case *lstm.Layer:
		return lstm.ApplyGradSGD(v, dW, lr)
	case *embedding.Layer:
		return embedding.ApplyGradSGD(v, dW, lr)
	case *sequential.Layer:
		return sequential.ApplyGradSGD(v, dW, lr)
	case *residual.Layer:
		return residual.ApplyGradSGD(v, dW, lr)
	case *Layer:
		return ApplyGradSGD(v, dW, lr)
	case *kmeans.Layer:
		return kmeans.ApplyGradSGD(v, dW, lr)
	case *mamba.Layer:
		return mamba.ApplyGradSGD(v, dW, lr)
	case *metacognition.Layer:
		return metacognition.ApplyGradSGD(v, dW, lr)
	case *gdn.Layer:
		return gdn.ApplyGradSGD(v, dW, lr)
	default:
		return fmt.Errorf("parallel: ApplyGradSGD unsupported branch %T", op)
	}
}

func branchSyncExec(op any, exec core.ExecConfig) {
	switch v := op.(type) {
	case *dense.Layer:
		v.Exec = exec
	case *mha.Layer:
		v.Exec = exec
		if v.Q != nil {
			v.Q.Exec = exec
		}
		if v.K != nil {
			v.K.Exec = exec
		}
		if v.V != nil {
			v.V.Exec = exec
		}
		if v.O != nil {
			v.O.Exec = exec
		}
	case *swiglu.Layer:
		v.Exec = exec
		if v.Gate != nil {
			v.Gate.Exec = exec
		}
		if v.Up != nil {
			v.Up.Exec = exec
		}
		if v.Down != nil {
			v.Down.Exec = exec
		}
	case *rmsnorm.Layer:
		v.Exec = exec
	case *layernorm.Layer:
		v.Exec = exec
	case *softmax.Layer:
		v.Exec = exec
	case *cnn1.Layer:
		v.Exec = exec
		if v.Proj != nil {
			v.Proj.Exec = exec
		}
	case *cnn2.Layer:
		v.Exec = exec
		if v.Proj != nil {
			v.Proj.Exec = exec
		}
	case *cnn3.Layer:
		v.Exec = exec
		if v.Proj != nil {
			v.Proj.Exec = exec
		}
	case *convt1.Layer:
		v.Exec = exec
		if v.Proj != nil {
			v.Proj.Exec = exec
		}
	case *convt2.Layer:
		v.Exec = exec
		if v.Proj != nil {
			v.Proj.Exec = exec
		}
	case *convt3.Layer:
		v.Exec = exec
		if v.Proj != nil {
			v.Proj.Exec = exec
		}
	case *rnn.Layer:
		v.Exec = exec
	case *lstm.Layer:
		v.Exec = exec
	case *embedding.Layer:
		v.Exec = exec
	case *sequential.Layer:
		v.Exec = exec
		for _, ch := range v.Children {
			if ch != nil {
				ch.Exec = exec
			}
		}
	case *residual.Layer:
		v.Exec = exec
		for _, ch := range v.Children {
			if ch != nil {
				ch.Exec = exec
			}
		}
	case *Layer:
		v.Exec = exec
		v.SyncBranchExec()
	case *kmeans.Layer:
		v.Exec = exec
		if v.Centers != nil {
			v.Centers.Exec = exec
		}
	case *mamba.Layer:
		v.Exec = exec
	case *metacognition.Layer:
		v.Exec = exec
		if v.Observed != nil {
			v.Observed.Exec = exec
		}
	case *gdn.Layer:
		v.Exec = exec
	}
}
