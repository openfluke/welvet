package tokenizer

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/openfluke/welvet/entity"
)

// LoadForEntity resolves tokenizer.json for a mounted .entity.
// Order: embedded blob → header path → snapshot → sidecars → hub by repo.
func LoadForEntity(entityPath, headerTok, snapshot, repo string) (*Tokenizer, error) {
	if data, err := loadEmbedded(entityPath); err == nil && len(data) > 0 {
		tok, err := NewTokenizerFromJSON(data)
		if err == nil {
			return tok, nil
		}
	}

	var tried []string
	for _, p := range candidatePaths(entityPath, headerTok, snapshot, repo) {
		tried = append(tried, p)
		if st, err := os.Stat(p); err != nil || st.IsDir() {
			continue
		}
		tok, err := LoadTokenizer(p)
		if err == nil {
			return tok, nil
		}
	}
	if len(tried) == 0 {
		return nil, fmt.Errorf("no tokenizer near %s (re-pull from Octo or re-convert with embedded tokenizer)", entityPath)
	}
	return nil, fmt.Errorf("no tokenizer near %s (tried %s)", entityPath, strings.Join(tried, ", "))
}

func loadEmbedded(entityPath string) ([]byte, error) {
	ef, err := entity.Open(entityPath)
	if err != nil {
		return nil, err
	}
	defer ef.Close()
	return ef.LoadTokenizerJSON()
}

func candidatePaths(entityPath, headerTok, snapshot, repo string) []string {
	dir := filepath.Dir(entityPath)
	stem := strings.TrimSuffix(filepath.Base(entityPath), filepath.Ext(entityPath))
	var out []string
	add := func(p string) {
		p = strings.TrimSpace(p)
		if p == "" {
			return
		}
		out = append(out, p)
	}

	add(filepath.Join(dir, stem+".tokenizer.json"))
	add(filepath.Join(dir, "tokenizer.json"))

	if headerTok != "" {
		add(headerTok)
		if !filepath.IsAbs(headerTok) {
			add(filepath.Join(dir, headerTok))
			if hub := os.Getenv("OCTO_HUB"); hub != "" {
				add(filepath.Join(filepath.Dir(hub), headerTok))
				add(filepath.Join(hub, strings.TrimPrefix(filepath.ToSlash(headerTok), "octo_hub/")))
			}
		}
	}
	if snapshot != "" {
		add(filepath.Join(snapshot, "tokenizer.json"))
		if !filepath.IsAbs(snapshot) {
			if hub := os.Getenv("OCTO_HUB"); hub != "" {
				add(filepath.Join(filepath.Dir(hub), snapshot, "tokenizer.json"))
			}
		}
	}
	if repo != "" {
		name := "models--" + strings.ReplaceAll(repo, "/", "--")
		rel := filepath.Join(name, "snapshots", "manual-download", "tokenizer.json")
		if hub := os.Getenv("OCTO_HUB"); hub != "" {
			add(filepath.Join(hub, rel))
		}
		add(filepath.Join("octo_hub", rel))
		add(filepath.Join(dir, "..", "octo_hub", rel))
	}
	return uniqueStrings(out)
}

func uniqueStrings(in []string) []string {
	seen := make(map[string]struct{}, len(in))
	out := make([]string, 0, len(in))
	for _, s := range in {
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	return out
}
