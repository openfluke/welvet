package transformer

import "strings"

// Turn is one user/assistant exchange.
type Turn struct {
	User      string
	Assistant string
}

// Template formats chat prompts (ChatML for SmolLM2).
type Template struct {
	Name         string
	RolePrefixes map[string]string
	RoleSuffixes map[string]string
	GlobalPrefix string
	GlobalSuffix string
}

// ChatML is used by SmolLM2 / Qwen-style instruct models.
var ChatML = Template{
	Name: "chatml",
	RolePrefixes: map[string]string{
		"system":    "<|im_start|>system\n",
		"user":      "<|im_start|>user\n",
		"assistant": "<|im_start|>assistant\n",
	},
	RoleSuffixes: map[string]string{
		"system":    "<|im_end|>\n",
		"user":      "<|im_end|>\n",
		"assistant": "<|im_end|>\n",
	},
}

// BuildPrompt constructs a full prompt from turns + current user message.
func (t Template) BuildPrompt(turns []Turn, systemPrompt, userMsg string) string {
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
