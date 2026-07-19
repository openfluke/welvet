package flux2

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"os"
)

// WritePNG writes an HxWx3 RGB uint8 buffer to path as a PNG.
func WritePNG(path string, rgb []uint8, height, width int) error {
	if height <= 0 || width <= 0 {
		return fmt.Errorf("WritePNG: bad size %dx%d", height, width)
	}
	if len(rgb) < height*width*3 {
		return fmt.Errorf("WritePNG: rgb short %d need %d", len(rgb), height*width*3)
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return EncodePNG(f, rgb, height, width)
}

// EncodePNG writes RGB HxWx3 uint8 to w.
func EncodePNG(w io.Writer, rgb []uint8, height, width int) error {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			off := (y*width + x) * 3
			img.SetRGBA(x, y, color.RGBA{
				R: rgb[off],
				G: rgb[off+1],
				B: rgb[off+2],
				A: 255,
			})
		}
	}
	return png.Encode(w, img)
}
