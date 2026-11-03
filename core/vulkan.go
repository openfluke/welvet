package core

import (
	"fmt"
	"sync"
)

// VulkanExecutor is installed by nativevulkan.EnableRouting. Keeping this contract
// in core avoids a layer ↔ native binding import cycle. All numerical work is
// performed by the independent native Vulkan backend.
type VulkanExecutor interface {
	TrainCredit(any, *Tensor[float32], *Tensor[float32], *Tensor[float32], int, float64) error
	DenseProxy(any, *Tensor[float32], *Tensor[float32], *Tensor[float32], int) (*Tensor[float32], error)
	Forward(any, *Tensor[float32]) (*Tensor[float32], *Tensor[float32], error)
	Backward(any, *Tensor[float32], *Tensor[float32]) (*Tensor[float32], *Tensor[float32], error)
	ApplySGD(any, *Tensor[float32], float64) error
	TrainMSE(any, *Tensor[float32], *Tensor[float32], int, float64) (float64, error)
}

var vulkanRoute struct {
	sync.RWMutex
	executor VulkanExecutor
}

// RegisterVulkanExecutor returns an unregister function. Only one router can own
// process-wide automatic placement at a time; explicit sessions are independent.
func RegisterVulkanExecutor(e VulkanExecutor) (func(), error) {
	vulkanRoute.Lock()
	defer vulkanRoute.Unlock()
	if e == nil || vulkanRoute.executor != nil {
		return nil, fmt.Errorf("vulkan: routing already registered or nil executor")
	}
	vulkanRoute.executor = e
	return func() {
		vulkanRoute.Lock()
		defer vulkanRoute.Unlock()
		if vulkanRoute.executor == e {
			vulkanRoute.executor = nil
		}
	}, nil
}
func VulkanRoutingAvailable() bool {
	vulkanRoute.RLock()
	defer vulkanRoute.RUnlock()
	return vulkanRoute.executor != nil
}
func vulkanInput[T Numeric](t *Tensor[T]) (*Tensor[float32], error) {
	v, ok := any(t).(*Tensor[float32])
	if !ok || v == nil {
		return nil, fmt.Errorf("vulkan: routing requires Float32 activation tensors")
	}
	return v, nil
}
func vulkanResult[T Numeric](t *Tensor[float32]) *Tensor[T] {
	if t == nil {
		return nil
	}
	return any(t).(*Tensor[T])
}
func VulkanForward[T Numeric](op any, x *Tensor[T]) (*Tensor[T], *Tensor[T], error) {
	v, err := vulkanInput(x)
	if err != nil {
		return nil, nil, err
	}
	vulkanRoute.RLock()
	defer vulkanRoute.RUnlock()
	if vulkanRoute.executor == nil {
		return nil, nil, fmt.Errorf("vulkan: call nativevulkan.EnableRouting before selecting BackendVulkan")
	}
	pre, post, err := vulkanRoute.executor.Forward(op, v)
	return vulkanResult[T](pre), vulkanResult[T](post), err
}
func VulkanBackward[T Numeric](op any, gy, x *Tensor[T]) (*Tensor[T], *Tensor[T], error) {
	v, err := vulkanInput(x)
	if err != nil {
		return nil, nil, err
	}
	g, err := vulkanInput(gy)
	if err != nil {
		return nil, nil, err
	}
	vulkanRoute.RLock()
	defer vulkanRoute.RUnlock()
	if vulkanRoute.executor == nil {
		return nil, nil, fmt.Errorf("vulkan: routing is not enabled")
	}
	gx, dw, err := vulkanRoute.executor.Backward(op, g, v)
	return vulkanResult[T](gx), vulkanResult[T](dw), err
}
func VulkanApplySGD[T Numeric](op any, dw *Tensor[T], lr float64) error {
	vulkanRoute.RLock()
	defer vulkanRoute.RUnlock()
	if vulkanRoute.executor == nil {
		return fmt.Errorf("vulkan: routing is not enabled")
	}
	if dw == nil || len(dw.Data) == 0 {
		return nil
	}
	v, err := vulkanInput(dw)
	if err != nil {
		return err
	}
	return vulkanRoute.executor.ApplySGD(op, v, lr)
}
func VulkanTrainMSE[T Numeric](op any, x, target *Tensor[T], mode int, lr float64) (float64, error) {
	v, err := vulkanInput(x)
	if err != nil {
		return 0, err
	}
	t, err := vulkanInput(target)
	if err != nil {
		return 0, err
	}
	vulkanRoute.RLock()
	defer vulkanRoute.RUnlock()
	if vulkanRoute.executor == nil {
		return 0, fmt.Errorf("vulkan: routing is not enabled")
	}
	return vulkanRoute.executor.TrainMSE(op, v, t, mode, lr)
}

func VulkanDenseProxy[T Numeric](op any, gy, x, pre *Tensor[T], kind int) (*Tensor[T], error) {
	v, err := vulkanInput(x)
	if err != nil {
		return nil, err
	}
	g, err := vulkanInput(gy)
	if err != nil {
		return nil, err
	}
	var p *Tensor[float32]
	if pre != nil {
		p, err = vulkanInput(pre)
		if err != nil {
			return nil, err
		}
	}
	vulkanRoute.RLock()
	defer vulkanRoute.RUnlock()
	if vulkanRoute.executor == nil {
		return nil, fmt.Errorf("vulkan: routing is not enabled")
	}
	out, err := vulkanRoute.executor.DenseProxy(op, g, v, p, kind)
	return vulkanResult[T](out), err
}

func VulkanTrainCredit[T Numeric](op any, gy, x, pre *Tensor[T], mode int, lr float64) error {
	v, err := vulkanInput(x)
	if err != nil {
		return err
	}
	g, err := vulkanInput(gy)
	if err != nil {
		return err
	}
	var p *Tensor[float32]
	if pre != nil {
		p, err = vulkanInput(pre)
		if err != nil {
			return err
		}
	}
	vulkanRoute.RLock()
	defer vulkanRoute.RUnlock()
	if vulkanRoute.executor == nil {
		return fmt.Errorf("vulkan: routing is not enabled")
	}
	return vulkanRoute.executor.TrainCredit(op, g, v, p, mode, lr)
}
