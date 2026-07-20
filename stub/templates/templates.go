package templates

import (
	"strings"
)

// Turn represents a single turn in a chat conversation.
type Turn struct {
	User      string
	Assistant string
}

// Template defines the formatting markers for different chat styles.
type Template struct {
	Name         string
	RolePrefixes map[string]string
	RoleSuffixes map[string]string
	GlobalPrefix string
	GlobalSuffix string
}

// BuildPrompt constructs a full prompt string from conversation turns.
func (t Template) BuildPrompt(turns []Turn, systemPrompt string, userMsg string) string {
	var sb strings.Builder
	sb.WriteString(t.GlobalPrefix)

	if systemPrompt != "" {
		if pre, ok := t.RolePrefixes["system"]; ok {
			sb.WriteString(pre)
			sb.WriteString(strings.TrimSpace(systemPrompt))
			sb.WriteString(t.RoleSuffixes["system"])
		}
	}

	for _, turn := range turns {
		if pre, ok := t.RolePrefixes["user"]; ok {
			sb.WriteString(pre)
			sb.WriteString(turn.User)
			sb.WriteString(t.RoleSuffixes["user"])
		}
		if pre, ok := t.RolePrefixes["assistant"]; ok {
			sb.WriteString(pre)
			sb.WriteString(turn.Assistant)
			sb.WriteString(t.RoleSuffixes["assistant"])
		}
	}

	if pre, ok := t.RolePrefixes["user"]; ok {
		sb.WriteString(pre)
		sb.WriteString(userMsg)
		sb.WriteString(t.RoleSuffixes["user"])
	}

	if pre, ok := t.RolePrefixes["assistant"]; ok {
		sb.WriteString(pre)
	}

	sb.WriteString(t.GlobalSuffix)
	return sb.String()
}

// BuildNextTurnSegment returns only the text that is NEW compared to KV cache.
func (t Template) BuildNextTurnSegment(userMsg string) string {
	var sb strings.Builder
	if pre, ok := t.RolePrefixes["user"]; ok {
		sb.WriteString(pre)
		sb.WriteString(userMsg)
		sb.WriteString(t.RoleSuffixes["user"])
	}
	if pre, ok := t.RolePrefixes["assistant"]; ok {
		sb.WriteString(pre)
	}
	return sb.String()
}

// Preset templates.
var (
	ChatML = Template{
		Name: "chatml",
		RolePrefixes: map[string]string{
			"system":    "<|im_start|>system\n",
			"user":      "<|im_start|>user\n",
			"assistant": "<|im_start|>assistant\n",
		},
		RoleSuffixes: map[string]string{
			"system":    "\n",
			"user":      "\n",
			"assistant": "\n",
		},
	}

	PlainCompletion = Template{
		Name:         "plain",
		RolePrefixes: map[string]string{"user": "", "assistant": ""},
		RoleSuffixes: map[string]string{"user": "", "assistant": ""},
	}

	BitNetInstruction = Template{
		Name: "bitnet-inst",
		RolePrefixes: map[string]string{
			"user":      "<s>[INST] ",
			"assistant": " ",
		},
		RoleSuffixes: map[string]string{
			"user":      " [/INST]",
			"assistant": " </s>",
		},
	}

	MicrosoftBitNetChat = Template{
		Name: "microsoft-bitnet",
		RolePrefixes: map[string]string{
			"system": "System: ", "user": "User: ", "assistant": "Assistant: ",
		},
		RoleSuffixes: map[string]string{
			"system": "<|eot_id|>", "user": "<|eot_id|>", "assistant": "<|eot_id|>",
		},
	}

	Llama3 = Template{
		Name: "llama3",
		RolePrefixes: map[string]string{
			"system":    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
			"user":      "<|start_header_id|>user<|end_header_id|>\n\n",
			"assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n",
		},
		RoleSuffixes: map[string]string{
			"system": "<|eot_id|>", "user": "<|eot_id|>", "assistant": "<|eot_id|>",
		},
	}
)
