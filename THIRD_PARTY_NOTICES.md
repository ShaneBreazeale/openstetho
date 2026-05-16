# Third-Party Notices

This repository's source code is Apache-2.0. Packaged binaries and app
bundles also include third-party Rust and Python dependencies under their
own licenses.

## Rust dependencies

The locked Rust dependency graph is predominantly permissive:
Apache-2.0, MIT, BSD-2-Clause/BSD-3-Clause, ISC, Zlib, Unicode-3.0,
0BSD, Unlicense, BSL-1.0, CC0-1.0, and Open Font License terms. The UI
stack may include default fonts from `epaint_default_fonts` under OFL
and related font licenses.

Before distributing a compiled binary, generate and ship a complete
notice bundle from the exact `Cargo.lock` used for the release. One
reasonable workflow is:

```bash
cargo install cargo-about
cargo about generate about.hbs > THIRD_PARTY_LICENSES.html
```

Review the generated output before release. Do not ship dependencies
with GPL/AGPL-only, proprietary, unknown, or source-unavailable terms
unless you have confirmed compatibility for the intended distribution.

## Python dependencies

The model pipeline depends on PyTorch, torchaudio, coremltools, librosa,
soundfile, numpy, scikit-learn, pandas, tqdm, matplotlib, pytest, and
ruff. These packages are not vendored in this repository. If you ship a
Python environment, wheelhouse, container image, or packaged model
toolchain, include the notices required by those package versions.
