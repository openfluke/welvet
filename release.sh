#!/usr/bin/env bash
# Auto-release openfluke/welvet for the current scorecard version.
#
# What it does:
#   1. Read version from ./README.md (scorecard)
#   2. Smoke-build the Go module (optional skip)
#   3. Commit engine source if dirty (nested w2a / book / apps stay gitignored)
#   4. Push main → origin
#   5. Tag + GitHub Release (create or update)
#
# Usage:
#   ./release.sh                 # full release
#   ./release.sh --dry-run       # version + smoke only
#   ./release.sh --no-push       # commit locally, skip push/release
#   ./release.sh --skip-build    # skip go build smoke
#   ./release.sh --retag         # force-move existing tag to HEAD + refresh release
#
# Sibling releases (optional, after this one):
#   cd w2a && ./release.sh                 # suite logs
#   cd openfluke.github.io && ./release.sh # feature book PDF
#
# Needs: git, go, and either gh (authenticated) or GITHUB_TOKEN.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

REPO_SLUG="openfluke/welvet"
API="https://api.github.com/repos/${REPO_SLUG}"

DRY_RUN=0
NO_PUSH=0
SKIP_BUILD=0
RETAG=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --no-push) NO_PUSH=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --retag) RETAG=1 ;;
    -h|--help)
      sed -n '2,28p' "$0"
      exit 0
      ;;
    *)
      echo "unknown flag: $arg" >&2
      exit 2
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Version from README.md scorecard
# ---------------------------------------------------------------------------
read_version() {
  python3 - <<'PY'
from pathlib import Path
import re
text = Path("README.md").read_text(encoding="utf-8")
earned = None
m = re.search(r"\*\*(\d+(?:\.\d+)?)\s*/\s*100\*\*\s*pts", text)
if m:
    earned = float(m.group(1))
if earned is None:
    m = re.search(r"\|\s*\*\*Version\*\*\s*\|\s*\*\*(v[\d.]+)\*\*", text)
    if m:
        v = m.group(1)
        earned = 100.0 if v == "v1.0" else float(v[3:]) if v.startswith("v0.") else None
if earned is None:
    raise SystemExit("could not parse Welvet version from README.md")
ver = "v1.0" if earned >= 100 else f"v0.{int(round(earned)):02d}"
print(f"{ver} {earned}")
PY
}

have_gh() { command -v gh >/dev/null 2>&1; }

token() {
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    echo "$GITHUB_TOKEN"
  elif [[ -n "${GH_TOKEN:-}" ]]; then
    echo "$GH_TOKEN"
  else
    echo ""
  fi
}

need_publish_tools() {
  if have_gh; then
    return 0
  fi
  if [[ -n "$(token)" ]]; then
    return 0
  fi
  echo "ERROR: need GitHub CLI (gh) or GITHUB_TOKEN to publish a release." >&2
  echo "  install: https://cli.github.com/  then  gh auth login" >&2
  echo "  or:      export GITHUB_TOKEN=ghp_…" >&2
  exit 1
}

smoke_build() {
  echo "→ go build (engine packages smoke)…"
  # Explicit roots — apps/w2a/book may exist on disk but are gitignored / not engine.
  # replace directives need sibling webgpu checkout.
  local roots=(
    ./core/... ./weights/... ./quant/... ./simd/... ./webgpu/... ./tiling/...
    ./architecture/... ./layers/... ./runtime/... ./systems/... ./model/...
    ./fusedgpu/... ./stub/...
  )
  if ! go build "${roots[@]}"; then
    echo "ERROR: go build failed. Fix compile errors or pass --skip-build." >&2
    exit 1
  fi
  echo "→ build OK"
}

release_exists() {
  local tag="$1"
  if have_gh; then
    gh release view "$tag" --repo "$REPO_SLUG" >/dev/null 2>&1
  else
    local code
    code=$(curl -sS -o /dev/null -w "%{http_code}" \
      -H "Authorization: Bearer $(token)" \
      -H "Accept: application/vnd.github+json" \
      "${API}/releases/tags/${tag}")
    [[ "$code" == "200" ]]
  fi
}

create_or_update_release() {
  local tag="$1"
  local earned="$2"
  local notes
  notes="$(cat <<EOF
## Welvet ${tag}

Pure Go AI engine release aligned to scorecard **${earned}/100** → **${tag}**.

### Module
\`\`\`bash
go get github.com/openfluke/welvet@${tag}
\`\`\`

### What's in this tree
- Layers, 34 dtypes, 20 quant formats
- Backends: CPU tiled · Plan 9 SIMD · WebGPU
- Native in-dtype SGD / storage-truth train (no retained f32 master)
- Cross-numeric train proven in w2a Step

### Related releases
- Validation suite: https://github.com/openfluke/w2a/releases
- Feature book: https://github.com/openfluke/openfluke.github.io/releases
- Showcase: https://github.com/openfluke/down-the-dem

### Site
https://openfluke.github.io/welvet/

### Regenerate
\`\`\`bash
cd welvet && ./release.sh
\`\`\`
EOF
)"

  if have_gh; then
    if release_exists "$tag"; then
      echo "  updating existing release ${tag}…"
      gh release edit "$tag" --repo "$REPO_SLUG" \
        --title "Welvet ${tag}" \
        --notes "$notes"
    else
      echo "  creating release ${tag}…"
      # Source-only release (Go module tag). No binary assets required.
      gh release create "$tag" --repo "$REPO_SLUG" \
        --title "Welvet ${tag}" \
        --notes "$notes"
    fi
    return
  fi

  local auth="Authorization: Bearer $(token)"
  if release_exists "$tag"; then
    local id
    id=$(curl -sS -H "$auth" -H "Accept: application/vnd.github+json" \
      "${API}/releases/tags/${tag}" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
    local body
    body=$(VERSION="$tag" NOTES="$notes" python3 - <<'PY'
import json, os
print(json.dumps({
  "name": f"Welvet {os.environ['VERSION']}",
  "body": os.environ["NOTES"],
  "draft": False,
  "prerelease": False,
}))
PY
)
    curl -sS -X PATCH -H "$auth" -H "Accept: application/vnd.github+json" \
      -H "Content-Type: application/json" \
      -d "$body" "${API}/releases/${id}" >/dev/null
  else
    local body
    body=$(VERSION="$tag" NOTES="$notes" python3 - <<'PY'
import json, os
print(json.dumps({
  "tag_name": os.environ["VERSION"],
  "name": f"Welvet {os.environ['VERSION']}",
  "body": os.environ["NOTES"],
  "draft": False,
  "prerelease": False,
}))
PY
)
    curl -sS -H "$auth" -H "Accept: application/vnd.github+json" \
      -H "Content-Type: application/json" \
      -d "$body" "${API}/releases" >/dev/null
  fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if [[ ! -f "$ROOT/README.md" ]]; then
  echo "ERROR: README.md not found in $ROOT" >&2
  exit 1
fi

read -r VERSION EARNED <<<"$(read_version)"

echo "════════════════════════════════════════"
echo " Welvet engine release"
echo " version:  ${VERSION}  (${EARNED}/100)"
echo " repo:     ${REPO_SLUG}"
echo "════════════════════════════════════════"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  smoke_build
else
  echo "→ --skip-build: skipping go build"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo ""
  echo "dry-run: skipping commit / push / release"
  echo "would release: ${VERSION}"
  exit 0
fi

echo ""
echo "→ git status"
git status --short || true

if [[ -n "$(git status --porcelain)" ]]; then
  echo "→ committing engine source…"
  git add -A
  # belt + suspenders — never stage nested / generated trees
  git reset -q -- w2a openfluke.github.io apps dist .cache .gotmp 2>/dev/null || true
  if [[ -n "$(git diff --cached --name-only)" ]]; then
    git commit -m "$(cat <<EOF
Release Welvet ${VERSION}

Scorecard ${EARNED}/100. Engine source tagged for Go module consumers.
EOF
)"
  else
    echo "→ nothing staged — skip commit"
  fi
else
  echo "→ working tree clean — nothing to commit"
fi

if [[ "$NO_PUSH" -eq 1 ]]; then
  echo "→ --no-push: skipping push + GitHub release"
  echo "  tag when ready: git tag -a ${VERSION} && git push origin ${VERSION}"
  exit 0
fi

need_publish_tools

echo "→ pushing main…"
git push origin HEAD

echo "→ publishing GitHub Release ${VERSION}…"
if git rev-parse "$VERSION" >/dev/null 2>&1; then
  if [[ "$RETAG" -eq 1 ]]; then
    echo "  --retag: moving tag ${VERSION} → HEAD"
    git tag -d "$VERSION"
    git tag -a "$VERSION" -m "Welvet ${VERSION} (scorecard ${EARNED}/100)"
    git push origin "refs/tags/${VERSION}" --force
  else
    echo "  tag ${VERSION} already exists locally (pass --retag to move it to HEAD)"
    # still try to push in case remote is missing it
    git push origin "$VERSION" 2>/dev/null || git push origin "refs/tags/${VERSION}" || true
  fi
else
  git tag -a "$VERSION" -m "Welvet ${VERSION} (scorecard ${EARNED}/100)"
  git push origin "$VERSION" 2>/dev/null || git push origin "refs/tags/${VERSION}"
fi

create_or_update_release "$VERSION" "$EARNED"

echo ""
echo "════════════════════════════════════════"
echo " Done · ${VERSION}"
echo " Repo:    https://github.com/${REPO_SLUG}"
echo " Release: https://github.com/${REPO_SLUG}/releases/tag/${VERSION}"
echo " Module:  go get github.com/openfluke/welvet@${VERSION}"
echo "════════════════════════════════════════"
echo ""
echo "Next (optional):"
echo "  cd w2a && ./release.sh"
echo "  cd openfluke.github.io && ./release.sh"
