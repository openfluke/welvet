package flux2

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/openfluke/welvet/tokenizer"
)

// Klein Qwen3 prompt-embed extraction (diffusers Flux2KleinPipeline._get_qwen3_prompt_embeds):
//
//  1. Chat-template the prompt (role=user, add_generation_prompt, enable_thinking=False).
//  2. Tokenize / pad / truncate to max_sequence_length (default 512).
//  3. Forward Qwen3ForCausalLM with output_hidden_states=True.
//  4. Stack hidden_states[k] for k in (9, 18, 27) → [B, 3, S, 2560].
//  5. Permute + reshape → [B, S, 7680] (= joint_attention_dim).
//
// text_encoder-mlx-4bit uses MLX AffineQuantized 4-bit g64 (see quant.FormatAffinePacked).

// DefaultKleinHiddenLayers are HF hidden_states indices (embed=0) concatenated for Klein embeds.
// Diffusers: stack(output.hidden_states[k] for k in (9,18,27)). That is after layers 8,17,26.
var DefaultKleinHiddenLayers = []int{9, 18, 27}

const (
	// Qwen3KleinHiddenSize is Qwen3-4B hidden_size.
	Qwen3KleinHiddenSize = 2560
	// DefaultMaxPromptTokens matches Flux2KleinPipeline.tokenizer_max_length.
	DefaultMaxPromptTokens = 512
)

// ErrTextEncoderNotReady is returned when the text encoder directory exists but
// weight files are missing/incomplete.
var ErrTextEncoderNotReady = fmt.Errorf(
	"flux2 text encoder: weights incomplete under text_encoder-mlx-4bit/; " +
		"or supply precomputed embeds via LoadPromptEmbeds (raw float32 or .npy)",
)

// IsFlux2KleinPipeline reports whether snapshotDir/model_index.json is a Flux2KleinPipeline.
func IsFlux2KleinPipeline(snapshotDir string) bool {
	data, err := os.ReadFile(filepath.Join(snapshotDir, "model_index.json"))
	if err != nil {
		return false
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return false
	}
	name, _ := raw["_class_name"].(string)
	return name == "Flux2KleinPipeline"
}

// FindTokenizerDir returns tokenizer/ or text_encoder-mlx-4bit/ under the snapshot.
func FindTokenizerDir(snapshotDir string) (string, error) {
	candidates := []string{
		filepath.Join(snapshotDir, "tokenizer"),
		filepath.Join(snapshotDir, "text_encoder-mlx-4bit"),
	}
	for _, d := range candidates {
		tok := filepath.Join(d, "tokenizer.json")
		if st, err := os.Stat(tok); err == nil && !st.IsDir() {
			return d, nil
		}
	}
	return "", fmt.Errorf("no tokenizer.json under %s (expected tokenizer/ or text_encoder-mlx-4bit/)", snapshotDir)
}

// FindTextEncoderDir returns the MLX 4-bit text encoder directory when present.
func FindTextEncoderDir(snapshotDir string) (string, error) {
	d := filepath.Join(snapshotDir, "text_encoder-mlx-4bit")
	cfg := filepath.Join(d, "config.json")
	if st, err := os.Stat(cfg); err == nil && !st.IsDir() {
		return d, nil
	}
	return "", fmt.Errorf("no text_encoder-mlx-4bit/config.json under %s", snapshotDir)
}

// TokenizePrompt loads the snapshot tokenizer and encodes text (no chat template yet).
// Returns token ids truncated to maxLen (no padding).
func TokenizePrompt(snapshotDir, prompt string, maxLen int) ([]uint32, error) {
	if maxLen <= 0 {
		maxLen = DefaultMaxPromptTokens
	}
	dir, err := FindTokenizerDir(snapshotDir)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.LoadTokenizer(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	ids := tok.Encode(prompt, true)
	if len(ids) > maxLen {
		ids = ids[:maxLen]
	}
	return ids, nil
}

// EncodePrompt runs Qwen3 → Klein joint embeds [seq * jointAttentionDim] (7680).
//
// Uses chat template (enable_thinking=False), AffineQuantized 4-bit g64 weights under
// text_encoder-mlx-4bit/, and stacks HF hidden_states[9], [18], [27].
func EncodePrompt(snapshotDir, prompt string, maxLen int) (embeds []float32, txtSeq int, err error) {
	if _, err := FindTokenizerDir(snapshotDir); err != nil {
		return nil, 0, err
	}
	te, err := FindTextEncoderDir(snapshotDir)
	if err != nil {
		return nil, 0, err
	}
	if !textEncoderWeightsReady(te) {
		return nil, 0, fmt.Errorf("%w (text encoder dir present but weights incomplete under %s)", ErrTextEncoderNotReady, te)
	}
	return EncodePromptQwen(snapshotDir, prompt, maxLen)
}

func textEncoderWeightsReady(teDir string) bool {
	for _, name := range []string{"model.safetensors", "model.safetensors.index.json"} {
		p := filepath.Join(teDir, name)
		if st, err := os.Stat(p); err == nil && !st.IsDir() && st.Size() > 0 {
			return true
		}
	}
	return false
}

// LoadPromptEmbeds reads precomputed prompt embeds from path.
//
// Supported:
//   - NumPy .npy float16/float32, C-order, shape [S, D] or [1, S, D] (D = joint_attention_dim)
//   - Raw little-endian float32 blob of length S*D (D defaults to cfg joint dim / 7680)
//
// Returns flat [S*D] and sequence length S.
func LoadPromptEmbeds(path string, jointDim int) (embeds []float32, txtSeq int, err error) {
	if jointDim <= 0 {
		jointDim = DefaultConfig().JointAttentionDim
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	lower := strings.ToLower(path)
	if strings.HasSuffix(lower, ".npy") {
		return decodeNpyPromptEmbeds(data, jointDim)
	}
	if len(data)%4 != 0 {
		return nil, 0, fmt.Errorf("LoadPromptEmbeds: raw float32 file length %d not divisible by 4", len(data))
	}
	n := len(data) / 4
	if n%jointDim != 0 {
		return nil, 0, fmt.Errorf("LoadPromptEmbeds: %d floats not divisible by jointDim %d", n, jointDim)
	}
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out, n / jointDim, nil
}

func decodeNpyPromptEmbeds(data []byte, jointDim int) ([]float32, int, error) {
	if len(data) < 10 || string(data[:6]) != "\x93NUMPY" {
		return nil, 0, fmt.Errorf("LoadPromptEmbeds: not a .npy file")
	}
	major, minor := data[6], data[7]
	var headerLen int
	var headerOff int
	switch major {
	case 1:
		headerLen = int(binary.LittleEndian.Uint16(data[8:10]))
		headerOff = 10
	case 2:
		if len(data) < 12 {
			return nil, 0, fmt.Errorf("LoadPromptEmbeds: truncated npy v2 header")
		}
		headerLen = int(binary.LittleEndian.Uint32(data[8:12]))
		headerOff = 12
	default:
		return nil, 0, fmt.Errorf("LoadPromptEmbeds: unsupported npy version %d.%d", major, minor)
	}
	if headerOff+headerLen > len(data) {
		return nil, 0, fmt.Errorf("LoadPromptEmbeds: truncated npy header")
	}
	header := string(data[headerOff : headerOff+headerLen])
	payload := data[headerOff+headerLen:]

	descr, fortran, shape, err := parseNpyHeader(header)
	if err != nil {
		return nil, 0, err
	}
	if fortran {
		return nil, 0, fmt.Errorf("LoadPromptEmbeds: Fortran-order .npy not supported")
	}
	seq, dim, err := npySeqDim(shape, jointDim)
	if err != nil {
		return nil, 0, err
	}
	n := seq * dim
	out := make([]float32, n)
	switch {
	case strings.Contains(descr, "<f4"), descr == "|f4", descr == "f4":
		if len(payload) < n*4 {
			return nil, 0, fmt.Errorf("LoadPromptEmbeds: f32 payload short")
		}
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(payload[i*4:]))
		}
	case strings.Contains(descr, "<f2"), descr == "|f2", descr == "f2",
		strings.Contains(descr, ">f2"):
		be := strings.Contains(descr, ">f2")
		if len(payload) < n*2 {
			return nil, 0, fmt.Errorf("LoadPromptEmbeds: f16 payload short")
		}
		for i := 0; i < n; i++ {
			var u16 uint16
			if be {
				u16 = binary.BigEndian.Uint16(payload[i*2:])
			} else {
				u16 = binary.LittleEndian.Uint16(payload[i*2:])
			}
			out[i] = float16ToF32(u16)
		}
	default:
		return nil, 0, fmt.Errorf("LoadPromptEmbeds: unsupported npy dtype %q (want float32/float16)", descr)
	}
	if dim != jointDim {
		// Allow loading; caller may still run if dims match transformer config.
		_ = jointDim
	}
	return out, seq, nil
}

func parseNpyHeader(header string) (descr string, fortran bool, shape []int, err error) {
	// Header looks like: {'descr': '<f4', 'fortran_order': False, 'shape': (512, 7680), }
	descr = extractNpyString(header, "descr")
	if descr == "" {
		return "", false, nil, fmt.Errorf("LoadPromptEmbeds: missing descr in npy header")
	}
	if strings.Contains(header, "'fortran_order': True") || strings.Contains(header, `"fortran_order": True`) {
		fortran = true
	}
	shape, err = extractNpyShape(header)
	if err != nil {
		return "", false, nil, err
	}
	return descr, fortran, shape, nil
}

func extractNpyString(header, key string) string {
	for _, quote := range []string{"'", `"`} {
		needle := quote + key + quote + ":"
		i := strings.Index(header, needle)
		if i < 0 {
			continue
		}
		rest := header[i+len(needle):]
		rest = strings.TrimSpace(rest)
		if len(rest) == 0 {
			continue
		}
		q := rest[0]
		if q != '\'' && q != '"' {
			continue
		}
		rest = rest[1:]
		j := strings.IndexByte(rest, q)
		if j < 0 {
			continue
		}
		return rest[:j]
	}
	return ""
}

func extractNpyShape(header string) ([]int, error) {
	i := strings.Index(header, "shape")
	if i < 0 {
		return nil, fmt.Errorf("LoadPromptEmbeds: missing shape in npy header")
	}
	rest := header[i:]
	l := strings.IndexByte(rest, '(')
	r := strings.IndexByte(rest, ')')
	if l < 0 || r < 0 || r <= l {
		return nil, fmt.Errorf("LoadPromptEmbeds: bad shape in npy header")
	}
	inner := strings.TrimSpace(rest[l+1 : r])
	if inner == "" {
		return nil, fmt.Errorf("LoadPromptEmbeds: empty shape")
	}
	parts := strings.Split(inner, ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		var n int
		if _, err := fmt.Sscanf(p, "%d", &n); err != nil {
			return nil, fmt.Errorf("LoadPromptEmbeds: bad shape dim %q", p)
		}
		out = append(out, n)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("LoadPromptEmbeds: empty shape")
	}
	return out, nil
}

func npySeqDim(shape []int, jointDim int) (seq, dim int, err error) {
	switch len(shape) {
	case 1:
		if shape[0]%jointDim != 0 {
			return 0, 0, fmt.Errorf("LoadPromptEmbeds: 1D shape %v not divisible by jointDim %d", shape, jointDim)
		}
		return shape[0] / jointDim, jointDim, nil
	case 2:
		return shape[0], shape[1], nil
	case 3:
		// [B, S, D] — require B==1
		if shape[0] != 1 {
			return 0, 0, fmt.Errorf("LoadPromptEmbeds: batch dim %d != 1", shape[0])
		}
		return shape[1], shape[2], nil
	default:
		return 0, 0, fmt.Errorf("LoadPromptEmbeds: unsupported shape %v", shape)
	}
}

func float16ToF32(u uint16) float32 {
	sign := uint32(u>>15) & 1
	exp := int((u >> 10) & 0x1f)
	frac := uint32(u & 0x3ff)
	var fBits uint32
	switch exp {
	case 0:
		if frac == 0 {
			fBits = sign << 31
		} else {
			e := -14
			for frac&0x400 == 0 {
				frac <<= 1
				e--
			}
			frac &= 0x3ff
			fBits = (sign << 31) | (uint32(e+127) << 23) | (frac << 13)
		}
	case 31:
		fBits = (sign << 31) | (0xff << 23) | (frac << 13)
	default:
		fBits = (sign << 31) | (uint32(exp-15+127) << 23) | (frac << 13)
	}
	return math.Float32frombits(fBits)
}
