# Evaluation covariance assets (not production)

24 small legacy NPZ files, approximately 200 KB total: 12 for the historical
September 11–12 stress test and 12 for the September 18–19 released-target
diagnostic. They contain covariance parameters, not competition observations,
labels or saved predictions. They never replace `models/current_baseline`.

**Original training provenance is incomplete.** `manifest.json` explicitly
records `creation_code_version` and `creation_input_hashes` as unknown. It is
not a certificate that these files were fitted with today's code.

At migration, the training fields were reconstructed from the documented
date-excluded recipes. For the recent diagnostic their digest matches the
existing key and a legacy fitting log corroborates the member, heldout
recording and field sizes. The historical filenames store no field digest.
Evidence files are referenced with hashes; the complete old logs are not
required for inference and are not bundled. Successful loading or a filename
alone is not treated as provenance evidence.

The manifest seals current input, GP code, settings, package versions, full
reconstructed training-field hashes and asset bytes. Future runs check every
one of these before loading. A mismatch stops; it never triggers training.
Inference and other-task code hashes/configuration are recorded in each run.

Runs using these assets require `--allow-legacy-provenance` and keep their
results labeled conditional on legacy covariance provenance. Without that
flag forecast evaluation fails closed. Baseline nonforecast evaluation needs
no such acknowledgement. Do not edit a manifest to bypass a mismatch.

These assets and their manifest are intended to be tracked along with source.
Raw inputs remain separately authorized, untracked files. See
`docs/local_evaluation.md` for the supported commands, limitations and policy
questions. No automatic refit command is supplied.
