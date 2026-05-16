#!/usr/bin/env bash
#
# Download PhysioNet CirCor DigiScope Phonocardiogram Dataset (v1.0.3).
# ~3.3 GB, public, Open Data Commons Attribution license.
#
# Saves to data/circor/ (gitignored).
#
# Usage:
#   bash scripts/download_circor.sh
#
# Run from the repo root. Idempotent — wget -c / curl -C - resumes on interrupt.

set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data/circor"
mkdir -p "$DEST"

BASE_URL="https://physionet.org/files/circor-heart-sound/1.0.3/"

if command -v wget >/dev/null 2>&1; then
    echo "→ using wget"
    echo "→ downloading CirCor 1.0.3 to $DEST"
    cd "$DEST"
    wget -r -N -c -np -nH --cut-dirs=4 \
        --reject "index.html*,robots.txt,*.gif" \
        "$BASE_URL"
elif command -v curl >/dev/null 2>&1; then
    echo "→ wget not found; falling back to curl + a Python recursive crawler"
    echo "   (install wget with \`brew install wget\` for a faster download)"
    PY="${PYTHON:-python3}"
    if ! command -v "$PY" >/dev/null 2>&1; then
        echo "ERROR: neither wget nor python3 found; install one and retry" >&2
        exit 1
    fi
    "$PY" - "$BASE_URL" "$DEST" <<'PYEOF'
import os, sys, urllib.parse, urllib.request, html.parser, time
base, dest = sys.argv[1], sys.argv[2]

class L(html.parser.HTMLParser):
    def __init__(self): super().__init__(); self.hrefs=[]
    def handle_starttag(self, t, a):
        if t == "a":
            for k,v in a:
                if k == "href" and v: self.hrefs.append(v)

def index(url):
    with urllib.request.urlopen(url) as r:
        body = r.read().decode("utf-8", "ignore")
    p = L(); p.feed(body); return p.hrefs

def fetch_to(url, path):
    if os.path.exists(path):
        local = os.path.getsize(path)
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            remote = int(r.headers.get("Content-Length", "0"))
        if remote and local == remote:
            return
        # Resume
        req = urllib.request.Request(url, headers={"Range": f"bytes={local}-"})
        mode = "ab"
    else:
        req = urllib.request.Request(url); mode = "wb"
    with urllib.request.urlopen(req) as r, open(path, mode) as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk: break
            f.write(chunk)

visited = set()
queue = [base]
n_files = 0
while queue:
    url = queue.pop(0)
    if url in visited: continue
    visited.add(url)
    rel = url[len(base):]
    target_dir = os.path.join(dest, rel)
    os.makedirs(target_dir, exist_ok=True)
    print(f"  {rel or '/'}")
    for href in index(url):
        if href.startswith(("?", "#", "../")) or href.startswith("http"):
            continue
        if href.endswith("/"):
            queue.append(url + href)
        else:
            file_url = url + href
            file_path = os.path.join(target_dir, href)
            for attempt in range(3):
                try:
                    fetch_to(file_url, file_path)
                    n_files += 1
                    if n_files % 50 == 0:
                        print(f"    [{n_files} files]")
                    break
                except Exception as e:
                    print(f"    retry {href}: {e}")
                    time.sleep(2 * (attempt + 1))
            else:
                print(f"    FAILED {href}", file=sys.stderr)

print(f"\n✓ {n_files} files in {dest}")
PYEOF
else
    echo "ERROR: neither wget nor curl found" >&2
    exit 1
fi

echo
echo "✓ done. Tree summary:"
du -sh "$DEST"
echo
echo "Top-level entries:"
ls -la "$DEST" | head -20
