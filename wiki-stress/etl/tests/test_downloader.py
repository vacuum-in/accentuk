import hashlib
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import pytest

from ukstress.downloader import ChecksumMismatchError, download_dump, sha256_file


class _HTTPServer:
    def __init__(self, directory: Path) -> None:
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(directory))
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        host, port = cast(tuple[str, int], self.server.server_address)
        return f"http://{host}:{port}"

    def __exit__(self, *_: object) -> None:
        self.server.shutdown()
        self.thread.join()


def test_download_resumes_and_writes_manifest(tmp_path: Path) -> None:
    payload = b"Ukrainian Wiktionary dump bytes" * 128
    source = tmp_path / "source"
    source.mkdir()
    (source / "dump.bz2").write_bytes(payload)
    destination = tmp_path / "data" / "dump.bz2"
    destination.parent.mkdir()
    destination.with_suffix(".bz2.part").write_bytes(payload[:20])

    with _HTTPServer(source) as url:
        manifest = download_dump(
            f"{url}/dump.bz2", destination, hashlib.sha256(payload).hexdigest()
        )

    assert destination.read_bytes() == payload
    assert sha256_file(destination) == manifest.sha256
    assert destination.with_suffix(".bz2.manifest.json").exists()
    assert not destination.with_suffix(".bz2.part").exists()


def test_bad_checksum_is_not_accepted(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "dump.bz2").write_bytes(b"not the expected dump")
    destination = tmp_path / "dump.bz2"

    with _HTTPServer(source) as url, pytest.raises(ChecksumMismatchError):
        download_dump(f"{url}/dump.bz2", destination, "0" * 64)

    assert not destination.exists()
    assert not destination.with_suffix(".bz2.part").exists()
