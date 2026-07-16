package core

import "fmt"

// LayerType mirrors loom/poly layer kinds — each has a welvet/<name> package.
type LayerType int

const (
	LayerDense              LayerType = 0
	LayerMultiHeadAttention LayerType = 1
	LayerSwiGLU             LayerType = 2
	LayerRMSNorm            LayerType = 3
	LayerCNN1               LayerType = 4
	LayerCNN2               LayerType = 5
	LayerCNN3               LayerType = 6
	LayerRNN                LayerType = 7
	LayerLSTM               LayerType = 8
	LayerLayerNorm          LayerType = 9
	LayerConvTransposed1D   LayerType = 10
	LayerConvTransposed2D   LayerType = 11
	LayerConvTransposed3D   LayerType = 12
	LayerEmbedding          LayerType = 13
	LayerKMeans             LayerType = 14
	LayerSoftmax            LayerType = 15
	LayerParallel           LayerType = 16
	LayerSequential         LayerType = 17
	LayerResidual           LayerType = 18
	LayerMetacognition      LayerType = 19
)

func (t LayerType) String() string {
	switch t {
	case LayerDense:
		return "Dense"
	case LayerMultiHeadAttention:
		return "MultiHeadAttention"
	case LayerSwiGLU:
		return "SwiGLU"
	case LayerRMSNorm:
		return "RMSNorm"
	case LayerCNN1:
		return "CNN1"
	case LayerCNN2:
		return "CNN2"
	case LayerCNN3:
		return "CNN3"
	case LayerRNN:
		return "RNN"
	case LayerLSTM:
		return "LSTM"
	case LayerLayerNorm:
		return "LayerNorm"
	case LayerConvTransposed1D:
		return "ConvTransposed1D"
	case LayerConvTransposed2D:
		return "ConvTransposed2D"
	case LayerConvTransposed3D:
		return "ConvTransposed3D"
	case LayerEmbedding:
		return "Embedding"
	case LayerKMeans:
		return "KMeans"
	case LayerSoftmax:
		return "Softmax"
	case LayerParallel:
		return "Parallel"
	case LayerSequential:
		return "Sequential"
	case LayerResidual:
		return "Residual"
	case LayerMetacognition:
		return "Metacognition"
	default:
		return fmt.Sprintf("LayerType(%d)", int(t))
	}
}

// ActivationType is applied after linear maps (dense / projections).
type ActivationType int

const (
	ActivationLinear    ActivationType = -1
	ActivationReLU      ActivationType = 0
	ActivationSilu      ActivationType = 1
	ActivationGELU      ActivationType = 2
	ActivationTanh      ActivationType = 3
	ActivationSigmoid   ActivationType = 4
	ActivationLeakyReLU ActivationType = 5
	ActivationReLU2     ActivationType = 6
)

func (a ActivationType) String() string {
	switch a {
	case ActivationLinear:
		return "Linear"
	case ActivationReLU:
		return "ReLU"
	case ActivationSilu:
		return "SiLU"
	case ActivationGELU:
		return "GELU"
	case ActivationTanh:
		return "Tanh"
	case ActivationSigmoid:
		return "Sigmoid"
	case ActivationLeakyReLU:
		return "LeakyReLU"
	case ActivationReLU2:
		return "ReLU2"
	default:
		return fmt.Sprintf("Activation(%d)", int(a))
	}
}

// Backend selects the execution path for a layer call.
type Backend int

const (
	BackendCPUTiled Backend = 0
	BackendSIMD     Backend = 1
	BackendWebGPU   Backend = 2
)

func (b Backend) String() string {
	switch b {
	case BackendCPUTiled:
		return "cpu-tiled"
	case BackendSIMD:
		return "simd"
	case BackendWebGPU:
		return "webgpu"
	default:
		return fmt.Sprintf("Backend(%d)", int(b))
	}
}

// ExecConfig is shared by layer dispatchers.
type ExecConfig struct {
	Backend      Backend
	MultiCore    bool
	TileSize     int // 0 → layer default
	UseWebGPU    bool
	UseSIMD      bool
}
