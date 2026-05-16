#!/usr/bin/env bash
#
# Download PhysioNet/CinC Challenge 2016 — Classification of Heart Sound
# Recordings. 3125 PCG recordings across 6 subsets (training-a .. -f),
# captured with a mix of stethoscopes (some electronic). Public, free.
#
# Total uncompressed size: ~1.1 GB.
# License: Open Data Commons Attribution.
#
# Citation:
#   Liu C et al. "An open access database for the evaluation of heart
#   sound algorithms." Physiol Meas. 2016;37(12):2181-2213.
#
# Saves to data/physionet_2016/ (gitignored). Resumable.
#
# Usage:
#   bash scripts/download_physionet_2016.sh

set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data/physionet_2016"
mkdir -p "$DEST"
BASE_URL="https://physionet.org/files/challenge-2016/1.0.0/"

if command -v wget >/dev/null 2>&1; then
    echo "→ using wget"
    cd "$DEST"
    wget -r -N -c -np -nH --cut-dirs=4 \
        --reject "index.html*,robots.txt,*.gif" \
        "$BASE_URL"
elif command -v curl >/dev/null 2>&1; then
    echo "→ wget not found; using Python urllib fallback"
    echo "   (\`brew install wget\` for a faster download)"
    PY="${PYTHON:-python3}"
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
        return r.read().decode("utf-8", "ignore")

def fetch_to(url, path):
    if os.path.exists(path):
        local = os.path.getsize(path)
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            remote = int(r.headers.get("Content-Length", "0"))
        if remote and local == remote:
            return
        req = urllib.request.Request(url, headers={"Range": f"bytes={local}-"})
        mode = "ab"
    else:
        req = urllib.request.Request(url); mode = "wb"
    with urllib.request.urlopen(req) as r, open(path, mode) as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk: break
            f.write(chunk)

visited = set(); queue = [base]; n_files = 0
while queue:
    url = queue.pop(0)
    if url in visited: continue
    visited.add(url)
    rel = url[len(base):]
    target_dir = os.path.join(dest, rel)
    os.makedirs(target_dir, exist_ok=True)
    print(f"  {rel or '/'}")
    p = L()
    try:
        p.feed(index(url))
    except Exception as e:
        print(f"    index fail {rel}: {e}", file=sys.stderr); continue
    for href in p.hrefs:
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
                    if n_files % 100 == 0: print(f"    [{n_files} files]")
                    break
                except Exception as e:
                    print(f"    retry {href}: {e}"); time.sleep(2*(attempt+1))
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
ls -la "$DEST" | head -25
