package architecture

import "fmt"

func errCoord(which string, z, y, x, l int) error {
	return fmt.Errorf("architecture: bad %s coord z=%d y=%d x=%d l=%d", which, z, y, x, l)
}
