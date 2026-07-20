package entity

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
)

// NetworkDoc is a parsed network-only ENTITY header (volumetric grid).
type NetworkDoc struct {
	FormatVersion uint16
	Engine        string
	Status        string
	Network       json.RawMessage // PersistenceNetworkSpec JSON
	Blobs         []WeightBlob
	DataOffset    int
}

// SerializeNetwork builds ENTITY bytes from topology JSON + blob index + payload.
func SerializeNetwork(network any, blobs []WeightBlob, payload []byte) ([]byte, error) {
	netJSON, err := json.Marshal(network)
	if err != nil {
		return nil, fmt.Errorf("entity: marshal network: %w", err)
	}
	doc := headerDoc{
		FormatVersion: FormatVersion,
		Engine:        "welvet",
		Status:        "ok",
		Network:       netJSON,
		Blobs:         blobs,
	}
	headerJSON, err := json.Marshal(doc)
	if err != nil {
		return nil, fmt.Errorf("entity: marshal header: %w", err)
	}
	out := make([]byte, 0, fixedHeaderSize()+len(headerJSON)+len(payload))
	out = append(out, []byte(Magic)...)
	var ver [2]byte
	binary.LittleEndian.PutUint16(ver[:], FormatVersion)
	out = append(out, ver[:]...)
	out = append(out, 0, 0) // flags
	var hlen [8]byte
	binary.LittleEndian.PutUint64(hlen[:], uint64(len(headerJSON)))
	out = append(out, hlen[:]...)
	out = append(out, headerJSON...)
	out = append(out, payload...)
	return out, nil
}

// ParseNetwork reads a network-only ENTITY file (or any ENTITY with network section).
func ParseNetwork(data []byte) (*NetworkDoc, []byte, error) {
	if len(data) < fixedHeaderSize() {
		return nil, nil, fmt.Errorf("entity: file too short")
	}
	if string(data[:8]) != Magic {
		return nil, nil, fmt.Errorf("entity: bad magic %q", data[:8])
	}
	version := binary.LittleEndian.Uint16(data[8:10])
	if version != FormatVersion {
		return nil, nil, fmt.Errorf("entity: unsupported version %d", version)
	}
	headerLen := binary.LittleEndian.Uint64(data[12:20])
	if headerLen > headerMaxSize {
		return nil, nil, fmt.Errorf("entity: header too large")
	}
	dataOffset := fixedHeaderSize() + int(headerLen)
	if dataOffset > len(data) {
		return nil, nil, fmt.Errorf("entity: truncated header")
	}
	var doc headerDoc
	if err := json.Unmarshal(data[fixedHeaderSize():dataOffset], &doc); err != nil {
		return nil, nil, fmt.Errorf("entity: header JSON: %w", err)
	}
	if len(doc.Network) == 0 {
		return nil, nil, fmt.Errorf("entity: no network section")
	}
	return &NetworkDoc{
		FormatVersion: version,
		Engine:        doc.Engine,
		Status:        doc.Status,
		Network:       doc.Network,
		Blobs:         doc.Blobs,
		DataOffset:    dataOffset,
	}, data[dataOffset:], nil
}
