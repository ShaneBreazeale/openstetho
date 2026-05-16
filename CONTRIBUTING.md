# Contributing

Pull requests welcome. A few rules to keep the project's legal posture
intact.

## Do

- Submit code you wrote yourself, or clean-room re-implementations of
  algorithms documented in public literature with citations.
- Reference public datasets by URL; never commit dataset audio.
- Document any new protocol facts with the **device model + firmware
  version** you observed them on. State that you own the hardware.
- Add tests. The Rust workspace runs `cargo test`; the Python pipeline
  runs `uv run pytest model/tests`.

## Don't

- Commit any vendor's binary, decompiled source, or model weights.
- Commit dataset audio (CirCor, PASCAL, PhysioNet 2016, recordings of
  patients, etc.) even if you have rights to it personally.
- Commit your own captures of identifiable individuals without their
  written consent.
- Include vendor logos, brand colours, or other identifying marks in
  the UI or documentation. Mark names appear in text only, in their
  descriptive sense.
- Train a model using outputs from a proprietary teacher model and
  commit the resulting checkpoint. That work belongs in a private
  fork; it does not go into the public repository.

## Sign-off

By submitting a pull request, you certify that:

1. The contribution is your own original work, or is properly cited
   public-domain or compatibly-licensed work.
2. You have the right to submit it under the Apache License 2.0.
3. The contribution does not include vendor-proprietary material.

A simple `Signed-off-by: Your Name <you@example.com>` line in each
commit suffices.
