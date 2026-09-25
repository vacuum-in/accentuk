# Data governance and artifact policy

Every canonical derived record carries `source_id`, `utterance_id`, `license_id` (when
available), and a provenance mapping. Internal bootstrap/mining Parquet may retain an audio URI
and speaker identifier for reproducibility and leakage-safe splits; these fields are not public
training labels.

The text-resolver exporter omits audio references by default. Audio references are included only
when `TextResolverExportPolicy.include_audio_reference` is enabled and the source passes the
configured license policy (`redistribution`, commercial-use, and/or an allow-list of license IDs).
The exporter never copies audio bytes. JSONL/Parquet exports retain license and provenance fields,
and public text examples do not expose speaker IDs.

Media URIs and shard identifiers are validated as data, never interpolated into shell commands.
Paths containing control characters, remote authorities, or traversal components are rejected.
