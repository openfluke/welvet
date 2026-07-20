package donate

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
)

const (
	// DefaultPort sits beside construct TCP dev (17000).
	DefaultPort = 17001
	// MaxFrameBytes caps one framed JSON payload (model chunks use multiple frames).
	MaxFrameBytes = 64 << 20
)

// WriteFrame writes u32 little-endian length + JSON bytes to w.
func WriteFrame(w io.Writer, v any) error {
	body, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if len(body) > MaxFrameBytes {
		return errors.New("donate: frame exceeds MaxFrameBytes")
	}
	var hdr [4]byte
	binary.LittleEndian.PutUint32(hdr[:], uint32(len(body)))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	_, err = w.Write(body)
	return err
}

// ReadFrame reads one length-prefixed JSON object into dest (must be pointer).
func ReadFrame(r io.Reader, dest any) error {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return err
	}
	n := binary.LittleEndian.Uint32(hdr[:])
	if n == 0 || n > MaxFrameBytes {
		return errors.New("donate: invalid frame length")
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return err
	}
	return json.Unmarshal(buf, dest)
}

const (
	DonateComputeDefaultPort = DefaultPort
	MaxDonateFrameBytes      = MaxFrameBytes
)

func WriteDonateFrame(w io.Writer, v any) error { return WriteFrame(w, v) }
func ReadDonateFrame(r io.Reader, dest any) error { return ReadFrame(r, dest) }
