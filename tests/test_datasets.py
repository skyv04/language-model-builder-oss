"""Tests for model_builder.datasets.

Nothing here touches the network: downloads go through an in-memory fake of ``_urlopen`` (which
understands Range requests and can drop or stall connections) or through ``file://`` URLs.
"""

from __future__ import annotations

import dataclasses
import fcntl
import gzip
import hashlib
import http.client
import io
import json
import os
import random
import re
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

from model_builder import datasets, paths
from model_builder.datasets import _Recipe, _Remote

REAL_URLOPEN = datasets._urlopen
ICONS = {"moon", "book", "books", "globe", "leaf", "sparkles", "atom", "feather", "theater", "chat",
         "people", "list", "robot", "math", "pencil", "code", "question", "bubbles", "heart", "file",
         "scroll", "school"}
WORDS = ("apple river stone quiet bright garden window yellow music travel paper forest silver "
         "number cloud happy little orange morning winter castle market planet friend story").split()


def words(rng: random.Random, count: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(count))


def parquet_bytes(rows: list[dict], row_group_size: int | None = None) -> bytes:
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buffer, row_group_size=row_group_size)
    return buffer.getvalue()


def stories_text(count: int, seed: int = 1) -> tuple[bytes, list[str]]:
    rng = random.Random(seed)
    stories = [f"Once upon a time {words(rng, 60)}." for _ in range(count)]
    return ("\n<|endoftext|>\n".join(stories) + "\n<|endoftext|>\n").encode(), stories


class FakeResponse:
    def __init__(self, web: "FakeWeb", body: bytes, status: int, headers: dict, fail_at: int | None):
        self._web = web
        self._body = io.BytesIO(body)
        self.status = status
        self.headers = headers
        self._fail_at = fail_at

    def read(self, size: int = -1) -> bytes:
        if self._web.on_read is not None:
            self._web.on_read()
        if size is None or size < 0:
            size = len(self._body.getbuffer())
        if self._fail_at is not None:
            remaining = self._fail_at - self._body.tell()
            if remaining <= 0:
                raise http.client.IncompleteRead(b"")
            size = min(size, remaining)
        chunk = self._body.read(size)
        self._web.bytes_served += len(chunk)
        return chunk

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


class FakeWeb:
    """Stands in for ``datasets._urlopen``: serves ``files`` and honours Range headers."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.requests: list[tuple[str, dict]] = []
        self.ignore_range = False
        self.drop_after: dict[str, int] = {}  # url -> bytes after which its next response dies
        self.broken: set[str] = set()
        self.on_read = None
        self.bytes_served = 0

    def urlopen(self, request, timeout):
        url = request.full_url
        headers = {key.lower(): value for key, value in request.header_items()}
        self.requests.append((url, headers))
        if url in self.broken:
            raise ConnectionResetError("connection reset by the fake server")
        if url not in self.files:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, io.BytesIO())
        data = self.files[url]
        wanted = headers.get("range")
        if wanted and not self.ignore_range:
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", wanted)
            first = int(match.group(1))
            last = min(int(match.group(2)) if match.group(2) else len(data) - 1, len(data) - 1)
            if first >= len(data):
                raise urllib.error.HTTPError(url, 416, "Range Not Satisfiable", None, io.BytesIO())
            body = data[first:last + 1]
            headers = {"Content-Range": f"bytes {first}-{last}/{len(data)}",
                       "Content-Length": str(len(body))}
            return FakeResponse(self, body, 206, headers, self.drop_after.pop(url, None))
        return FakeResponse(self, data, 200, {"Content-Length": str(len(data))},
                            self.drop_after.pop(url, None))


class DatasetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="lmb-datasets-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.web = FakeWeb()
        for patcher in (
            mock.patch.dict(os.environ, {"LMB_DATA_DIR": str(self.root / "data")}),
            mock.patch.object(datasets, "_urlopen", self.web.urlopen),
            mock.patch.object(datasets, "RETRY_DELAY", 0.0),
            mock.patch.object(datasets, "_GUTENBERG_DELAY", 0.0),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def serve(self, name: str, data: bytes, checksum: bool = True) -> _Remote:
        url = f"https://files.example/{name}"
        self.web.files[url] = data
        return _Remote(url, len(data), hashlib.sha256(data).hexdigest() if checksum else None)

    def leftovers(self) -> list[str]:
        return sorted(path.name for path in paths.datasets_dir().iterdir()
                      if not path.name.endswith(".lock"))


class CatalogTests(DatasetTestCase):
    def test_entries_are_complete_and_json_serialisable(self) -> None:
        entries = datasets.catalog()
        self.assertEqual(len({entry.id for entry in entries}), len(entries))
        for kind in datasets.KINDS:
            subset = datasets.catalog(kind)
            self.assertGreaterEqual(len(subset), 8)
            self.assertEqual([entry.recommended for entry in subset].count(True), 1)
            self.assertTrue(subset[0].recommended)
            self.assertEqual(len({entry.icon for entry in subset}), len(subset))
            self.assertTrue(all(entry.kind == kind for entry in subset))
        self.assertEqual(datasets.catalog("pretrain")[0].id, "tinystories")
        self.assertEqual(datasets.catalog("finetune")[0].id, "everyday-conversations")
        for entry in entries:
            self.assertRegex(entry.id, r"^[a-z0-9][a-z0-9-]*$")
            self.assertIn(entry.icon, ICONS)
            self.assertRegex(entry.color, r"^#[0-9a-f]{6}$")
            self.assertTrue(entry.license and entry.license_url.startswith("https://"))
            self.assertTrue(entry.source_url.startswith("https://"))
            self.assertEqual(entry.paper_label is None, entry.paper_url is None)
            self.assertTrue(0 < entry.approx_bytes <= 60 * datasets.MB, entry.id)
            self.assertTrue(entry.name and entry.tagline and entry.description and entry.size_label)
            self.assertFalse(entry.custom)
            self.assertIs(datasets.get(entry.id), entry)
        json.dumps([dataclasses.asdict(entry) for entry in entries])

    def test_unknown_ids_and_kinds_are_rejected(self) -> None:
        for bad in ("nope", "../datasets", "custom-missing", "", "TinyStories"):
            with self.assertRaises(KeyError):
                datasets.get(bad)
        with self.assertRaises(ValueError):
            datasets.catalog("poetry")

    def test_state_before_download(self) -> None:
        self.assertFalse(datasets.status("tinystories")["downloaded"])
        self.assertEqual(datasets.status("tinystories")["documents"], 0)
        self.assertEqual(datasets.status("gsm8k")["rows"], 0)
        self.assertEqual(datasets.preview("tinystories"),
                         {"kind": "text", "documents": [], "truncated": False, "downloaded": False})
        self.assertEqual(datasets.preview("gsm8k"), {"kind": "chat", "examples": [], "downloaded": False})
        with self.assertRaises(datasets.DatasetNotDownloaded):
            datasets.iter_documents("tinystories")
        with self.assertRaises(ValueError):
            datasets.iter_documents("gsm8k")
        with self.assertRaises(ValueError):
            datasets.iter_conversations("tinystories")


class DownloadTests(DatasetTestCase):
    def test_download_normalises_text_and_writes_metadata(self) -> None:
        stories = ["the end of a story that was cut off.", "Once upon a time, Lily saw a ball.\r\nShe smiled.",
                   "Tom\u00a0had a  dog.   \n\n\n\nThe dog was big."]
        data = ("\n<|endoftext|>\n".join(stories) + "\n<|endoftext|>\n").encode()
        source = self.serve("stories.txt", data)
        calls = []
        with mock.patch.object(datasets, "_TINYSTORIES", source):
            result = datasets.download("tinystories", lambda *call: calls.append(call), threading.Event())
            self.assertEqual(list(datasets.iter_documents("tinystories")),
                             ["Once upon a time, Lily saw a ball.\nShe smiled.",
                              "Tom had a  dog.\n\nThe dog was big."])
            self.assertEqual((result["downloaded"], result["documents"]), (True, 2))
            folder = paths.datasets_dir() / "tinystories"
            self.assertEqual(sorted(path.name for path in folder.iterdir()), ["documents.jsonl", "meta.json"])
            body = (folder / "documents.jsonl").read_bytes()
            meta = json.loads((folder / "meta.json").read_text())
            self.assertEqual(meta["sha256"], hashlib.sha256(body).hexdigest())
            self.assertEqual((meta["bytes"], meta["documents"]), (len(body), 2))
            self.assertEqual((meta["sources"], meta["download_bytes"]), ([source.url], len(data)))
            self.assertEqual(meta["license"], "CDLA-Sharing-1.0")
            self.assertTrue(meta["created"].endswith("Z"))
            self.assertEqual(result["sha256"], meta["sha256"])
            self.assertEqual(calls[-1][0], calls[-1][1])
            self.assertTrue(all(total is None or done <= total for done, total, _ in calls))
            self.assertIn("LanguageModelBuilder", self.web.requests[0][1]["user-agent"])
            self.assertEqual(self.leftovers(), ["tinystories"])
            requests = len(self.web.requests)
            self.assertTrue(datasets.download("tinystories")["downloaded"])
            self.assertEqual(len(self.web.requests), requests)  # already there: nothing fetched
            datasets.delete_download("tinystories")
            self.assertFalse(datasets.status("tinystories")["downloaded"])
            self.assertEqual(self.leftovers(), [])

    def test_checksum_mismatch_leaves_nothing_behind(self) -> None:
        data, _ = stories_text(5)
        source = self.serve("stories.txt", data)
        with mock.patch.object(datasets, "_TINYSTORIES", _Remote(source.url, source.size, "0" * 64)):
            with self.assertRaisesRegex(datasets.DownloadError, "checksum"):
                datasets.download("tinystories")
        self.assertEqual(self.leftovers(), [])
        self.assertFalse(datasets.status("tinystories")["downloaded"])

    def test_dropped_connection_resumes_with_a_range_request(self) -> None:
        data, stories = stories_text(400)
        source = self.serve("stories.txt", data)
        self.web.drop_after[source.url] = 100_000
        with mock.patch.object(datasets, "_TINYSTORIES", source):
            self.assertEqual(datasets.download("tinystories")["documents"], len(stories))
        self.assertEqual([headers.get("range") for _, headers in self.web.requests], [None, "bytes=100000-"])

    def test_gives_up_after_repeated_network_errors(self) -> None:
        source = self.serve("stories.txt", stories_text(5)[0])
        self.web.broken.add(source.url)
        with mock.patch.object(datasets, "_TINYSTORIES", source):
            with self.assertRaisesRegex(datasets.DownloadError, "reset"):
                datasets.download("tinystories")
        self.assertEqual(len(self.web.requests), datasets.HTTP_RETRIES + 1)
        self.assertEqual(self.leftovers(), [])

    def test_http_errors_are_not_retried(self) -> None:
        missing = _Remote("https://files.example/missing.txt", 10, None)
        with mock.patch.object(datasets, "_TINYSTORIES", missing):
            with self.assertRaisesRegex(datasets.DownloadError, "404"):
                datasets.download("tinystories")
        self.assertEqual(len(self.web.requests), 1)
        self.assertEqual(self.leftovers(), [])

    def test_cancel_stops_the_download_and_removes_partial_files(self) -> None:
        source = self.serve("stories.txt", stories_text(1500)[0])
        cancel = threading.Event()
        self.web.on_read = cancel.set
        with mock.patch.object(datasets, "_TINYSTORIES", source):
            with self.assertRaises(datasets.DownloadCancelled):
                datasets.download("tinystories", None, cancel)
            self.assertLess(self.web.bytes_served, source.size)
            self.assertEqual(self.leftovers(), [])
            self.assertFalse(datasets.status("tinystories")["downloaded"])
            self.web.requests.clear()
            with self.assertRaises(datasets.DownloadCancelled):
                datasets.download("tinystories", cancel=cancel)
            self.assertEqual(self.web.requests, [])

    def test_second_download_of_the_same_dataset_is_refused(self) -> None:
        source = self.serve("stories.txt", stories_text(300)[0])
        reading, release = threading.Event(), threading.Event()
        errors: list[BaseException] = []

        def stall() -> None:
            reading.set()
            release.wait(10)

        def run() -> None:
            try:
                datasets.download("tinystories")
            except BaseException as error:  # pragma: no cover - reported below
                errors.append(error)

        self.web.on_read = stall
        with mock.patch.object(datasets, "_TINYSTORIES", source):
            worker = threading.Thread(target=run)
            worker.start()
            try:
                self.assertTrue(reading.wait(10))
                self.assertTrue(datasets.status("tinystories")["downloading"])
                with self.assertRaises(datasets.DownloadInProgress):
                    datasets.download("tinystories")
            finally:
                release.set()
                worker.join(30)
        self.assertEqual(errors, [])
        self.assertTrue(datasets.status("tinystories")["downloaded"])

    def test_lock_held_by_another_process_is_respected(self) -> None:
        with open(paths.datasets_dir() / ".tinystories.lock", "w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(datasets.DownloadInProgress, "another process"):
                datasets.download("tinystories")
        self.assertEqual(self.web.requests, [])

    def test_file_urls_work_with_the_real_opener(self) -> None:
        data, stories = stories_text(20)
        path = self.root / "stories.txt"
        path.write_bytes(data)
        source = _Remote(path.as_uri(), len(data), hashlib.sha256(data).hexdigest())
        with mock.patch.object(datasets, "_urlopen", REAL_URLOPEN), \
                mock.patch.object(datasets, "_TINYSTORIES", source):
            self.assertEqual(datasets.download("tinystories")["documents"], len(stories))
        self.assertEqual(list(datasets.iter_documents("tinystories")), stories)


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def system(text: str) -> dict:
    return {"role": "system", "content": text}


class RecipeTests(DatasetTestCase):
    def test_wikitext_rows_become_articles(self) -> None:
        rows = ["", " = Robert Boulter = \n", "", " Robert Boulter is an English actor . He was born in "
                "1 @,@ 984 . \n", "", " = = Career = = \n", "", " In 2000 he appeared in \" Taking the "
                "Ice \" . \n", " Note : Pos \n", " = Position ; GP = \n", " Games played \n", "",
                " = Second Article = \n", "", " It 's short ( really ) and does n't last . \n"]
        self.assertEqual(list(datasets._wikitext_articles(rows)), [
            "# Robert Boulter\n\nRobert Boulter is an English actor. He was born in 1,984.\n\n## Career"
            "\n\nIn 2000 he appeared in \"Taking the Ice\".\n\nNote: Pos = Position; GP = Games played",
            "# Second Article\n\nIt's short (really) and does n't last.",
        ])

    def test_gutenberg_boilerplate_is_removed(self) -> None:
        raw = ("\ufeffThe Project Gutenberg eBook of A Test\r\n\r\nThis ebook is for the use of anyone.\r\n"
               "\r\n*** START OF THE PROJECT GUTENBERG EBOOK A TEST ***\r\n\r\ncover\r\n\r\nProduced by A. "
               "Volunteer\r\n\r\n[Illustration: A cat]\r\n\r\nCHAPTER I.\r\n\r\nIt was a _very_ long line "
               "of prose that keeps going past the usual\r\nwrapping width, so it is joined to the next."
               "\r\n\r\n    Twinkle, twinkle, little star,\r\n    How I wonder what you are!\r\n\r\n*** END "
               "OF THE PROJECT GUTENBERG EBOOK A TEST ***\r\n\r\nSection 1. Terms of use.\r\n")
        self.assertEqual(datasets._strip_gutenberg(raw),
                         "CHAPTER I.\n\nIt was a very long line of prose that keeps going past the usual "
                         "wrapping width, so it is joined to the next.\n\n    Twinkle, twinkle, little "
                         "star,\n    How I wonder what you are!")

    def test_gutenberg_books_fall_back_to_the_second_mirror(self) -> None:
        rng = random.Random(3)

        def book(title: str) -> bytes:
            body = "\n\n".join(words(rng, 80) for _ in range(40))
            return (f"Header\n\n*** START OF THE PROJECT GUTENBERG EBOOK {title.upper()} ***\n\n{title}"
                    f"\n\n{body}\n\n*** END OF THE PROJECT GUTENBERG EBOOK {title.upper()} ***\n\n"
                    "Licence text.\n").encode()

        self.web.files["https://a.example/1.txt"] = book("First Book")
        self.web.files["https://b.example/2.txt"] = book("Second Book")
        self.web.files["https://a.example/3.txt"] = book("Short")[:200]
        mirrors = ("https://a.example/{0}.txt", "https://b.example/{0}.txt")
        books = ((1, "First Book", 1), (2, "Second Book", 1), (3, "Short", 1))
        with mock.patch.object(datasets, "_GUTENBERG_MIRRORS", mirrors), \
                mock.patch.object(datasets, "_GUTENBERG_BOOKS", books):
            self.assertEqual(datasets.download("gutenberg-classics")["documents"], 2)
        documents = list(datasets.iter_documents("gutenberg-classics"))
        self.assertEqual([text.split("\n\n")[0] for text in documents], ["First Book", "Second Book"])
        self.assertFalse(any("Licence" in text or "GUTENBERG" in text for text in documents))
        meta = json.loads((paths.datasets_dir() / "gutenberg-classics" / "meta.json").read_text())
        self.assertEqual(meta["sources"], ["https://a.example/1.txt", "https://b.example/2.txt",
                                           "https://a.example/3.txt"])

    def test_oasst_follows_the_best_ranked_replies(self) -> None:
        def message(mid, parent, role, text, rank=None, lang="en", deleted=False, review=True):
            return {"message_id": mid, "parent_id": parent, "text": text, "role": role, "lang": lang,
                    "review_result": review, "deleted": deleted, "rank": rank}

        rows = [
            message("r1", None, "prompter", "Hi there"),
            message("a1", "r1", "assistant", "Second best", rank=1),
            message("a2", "r1", "assistant", "Best answer", rank=0),
            message("p2", "a2", "prompter", "A follow-up nobody answered"),
            message("r2", None, "prompter", "Hallo", lang="de"),
            message("b2", "r2", "assistant", "Guten Tag", lang="de"),
            message("r3", None, "prompter", "Deleted question", deleted=True),
            message("b3", "r3", "assistant", "Orphaned answer"),
            message("r4", None, "prompter", "Only a rejected reply"),
            message("b4", "r4", "assistant", "Spam", review=False),
            message("r5", None, "prompter", "Tell me a joke"),
            message("c1", "r5", "assistant", "Unranked joke"),
            message("c2", "r5", "assistant", "Ranked joke", rank=0),
            message("c3", "c2", "prompter", "Another one"),
            message("c4", "c3", "assistant", "Third joke"),
        ]
        with mock.patch.object(datasets, "_OASST1", self.serve("oasst.parquet", parquet_bytes(rows))):
            self.assertEqual(datasets.download("oasst1")["rows"], 2)
        self.assertEqual([row["messages"] for row in datasets.iter_conversations("oasst1")], [
            [user("Hi there"), assistant("Best answer")],
            [user("Tell me a joke"), assistant("Ranked joke"), user("Another one"), assistant("Third joke")],
        ])

    def test_squad_questions_are_grouped_by_passage(self) -> None:
        def row(context: str, question: str, answer: str | None) -> dict:
            return {"id": question, "title": "T", "context": context, "question": question,
                    "answers": {"text": [answer] if answer else [], "answer_start": [0] if answer else []}}

        rows = [row("Passage A.", f"Question A{i}?", f"Answer A{i}") for i in range(7)]
        rows.insert(3, row("Passage A.", "Unanswerable?", None))
        rows.append(row("Passage B.", "Question B?", "Answer B"))
        with mock.patch.object(datasets, "_SQUAD", self.serve("squad.parquet", parquet_bytes(rows))):
            datasets.download("squad")
        conversations = [row["messages"] for row in datasets.iter_conversations("squad")]
        self.assertEqual([len(messages) for messages in conversations], [10, 4, 2])
        self.assertEqual(conversations[0][:3], [user("Passage A.\n\nQuestion A0?"), assistant("Answer A0"),
                                                user("Question A1?")])
        self.assertEqual(conversations[1][0], user("Passage A.\n\nQuestion A5?"))
        self.assertEqual(conversations[2], [user("Passage B.\n\nQuestion B?"), assistant("Answer B")])

    def test_gsm8k_answers_end_with_the_result(self) -> None:
        rows = [{"question": "Natalia sold 48 clips in April and half as many in May. How many in all?",
                 "answer": "In May she sold 48/2 = <<48/2=24>>24 clips.\nIn all she sold 48+24 = "
                           "<<48+24=72>>72 clips.\n#### 72"}]
        with mock.patch.object(datasets, "_GSM8K", self.serve("gsm8k.parquet", parquet_bytes(rows))):
            datasets.download("gsm8k")
        self.assertEqual(next(datasets.iter_conversations("gsm8k"))["messages"][1]["content"],
                         "In May she sold 48/2 = 24 clips.\nIn all she sold 48+24 = 72 clips.\n"
                         "The answer is 72.")

    def test_chat_parquet_rows_are_normalised(self) -> None:
        rows = [
            {"messages": [system("Be kind."), user(" Hi! "), assistant("Hello!\r\nHow are you?"), user("Bye")]},
            {"messages": [user("A question without a reply")]},
            {"messages": [user("Weather?"), {"role": "tool", "content": "{}"}, assistant("Sunny.")]},
        ]
        remote = self.serve("chat.parquet", parquet_bytes(rows))
        recipe = _Recipe(datasets._messages_recipe(remote))
        with mock.patch.dict(datasets._RECIPES, {"everyday-conversations": recipe}):
            self.assertEqual(datasets.download("everyday-conversations")["rows"], 2)
        self.assertEqual([row["messages"] for row in datasets.iter_conversations("everyday-conversations")],
                         [[system("Be kind."), user("Hi!"), assistant("Hello!\nHow are you?")],
                          [user("Weather?"), assistant("Sunny.")]])

    def test_arxiv_reads_a_truncated_multi_member_gzip(self) -> None:
        rng = random.Random(5)
        records = [{"id": str(i), "title": f"Paper  {i}\n about things",
                    "abstract": f"  {words(rng, 40)}\n more words  "} for i in range(400)]
        records[2]["abstract"] = "Too short."
        lines = [json.dumps(record).encode() + b"\n" for record in records]
        first, second = gzip.compress(b"".join(lines[:100])), gzip.compress(b"".join(lines[100:]))
        cut = len(first) + len(second) // 2
        remote = self.serve("arxiv.jsonl.gz", first + second, checksum=False)
        with mock.patch.object(datasets, "_ARXIV", remote), mock.patch.object(datasets, "_ARXIV_PREFIX", cut):
            datasets.download("arxiv-abstracts")
        documents = list(datasets.iter_documents("arxiv-abstracts"))
        expected = [f"Paper {record['id']} about things\n\n{' '.join(record['abstract'].split())}"
                    for record in records if record["abstract"] != "Too short."]
        self.assertTrue(99 < len(documents) < len(expected), len(documents))
        self.assertEqual(documents, expected[:len(documents)])
        self.assertEqual([headers.get("range") for _, headers in self.web.requests], [f"bytes=0-{cut - 1}"])

    def test_children_stories_read_a_cut_json_array(self) -> None:
        rng = random.Random(6)
        stories = [f"Once there was a {words(rng, 50)}." for _ in range(60)]
        items = [{"prompt": f"Write a story about {words(rng, 20)}", "text": story, "text_token_length": 99}
                 for story in stories]
        items.insert(1, {"prompt": "x", "text": "Too short."})
        data = json.dumps(items, indent=2).encode()
        remote = self.serve("children.json", data, checksum=False)
        with mock.patch.object(datasets, "_CHILDREN", remote), \
                mock.patch.object(datasets, "_CHILDREN_PREFIX", len(data) // 2):
            datasets.download("children-stories")
        documents = list(datasets.iter_documents("children-stories"))
        self.assertTrue(10 < len(documents) < len(stories), len(documents))
        self.assertEqual(documents, stories[:len(documents)])

    def partial_wikipedia(self) -> tuple[list[dict], bytes, list[str]]:
        rng = random.Random(7)
        rows = [{"id": str(i), "url": "https://example.org", "title": f"Article {i}",
                 "text": rng.randbytes(600).hex()} for i in range(2000)]
        data = parquet_bytes(rows, row_group_size=50)
        recipe = _Recipe(datasets._build_simple_wikipedia, max_bytes=100_000)
        with mock.patch.object(datasets, "_SIMPLE_WIKI", self.serve("wiki.parquet", data, checksum=False)), \
                mock.patch.dict(datasets._RECIPES, {"simple-wikipedia": recipe}):
            self.assertLessEqual(datasets.download("simple-wikipedia")["bytes"], 100_000)
        return rows, data, list(datasets.iter_documents("simple-wikipedia"))

    def test_partial_parquet_download_fetches_only_leading_row_groups(self) -> None:
        rows, data, documents = self.partial_wikipedia()
        self.assertGreater(len(documents), 50)
        self.assertEqual(documents, [f"Article {i}\n\n{rows[i]['text']}" for i in range(len(documents))])
        self.assertLess(self.web.bytes_served, len(data) // 4)
        self.assertTrue(all("range" in headers for _, headers in self.web.requests))

    def test_partial_parquet_download_copes_with_servers_ignoring_range(self) -> None:
        self.web.ignore_range = True
        rows, data, documents = self.partial_wikipedia()
        self.assertGreater(len(documents), 50)
        self.assertEqual(documents, [f"Article {i}\n\n{rows[i]['text']}" for i in range(len(documents))])


class PreviewTests(DatasetTestCase):
    def test_text_preview_is_capped(self) -> None:
        long_document = " ".join(["word"] * 3000)
        data = f"Short first document.\n\n\n{long_document}\n\n\nThird.".encode()
        info = datasets.import_custom("Long", "pretrain", "long.txt", data)
        result = datasets.preview(info.id)
        self.assertTrue(result["downloaded"] and result["truncated"])
        self.assertEqual(len(result["documents"]), 2)
        self.assertEqual(result["documents"][0], "Short first document.")
        self.assertTrue(result["documents"][1].endswith("word …"))
        self.assertLessEqual(sum(map(len, result["documents"])), datasets.PREVIEW_CHARS)

    def test_text_preview_without_spaces_stays_within_the_limit(self) -> None:
        info = datasets.import_custom("Blob", "pretrain", "blob.txt", b"x" * 20_000)
        result = datasets.preview(info.id)
        self.assertTrue(result["truncated"])
        self.assertLessEqual(len(result["documents"][0]), datasets.PREVIEW_CHARS)

    def test_chat_preview_shows_the_first_examples(self) -> None:
        rows = "\n".join(json.dumps({"prompt": f"Q{i}", "response": f"A{i}"}) for i in range(5))
        info = datasets.import_custom("Chats", "finetune", "chats.jsonl", rows.encode())
        self.assertEqual(datasets.preview(info.id), {
            "kind": "chat", "downloaded": True,
            "examples": [{"messages": [user(f"Q{i}"), assistant(f"A{i}")]} for i in range(3)],
        })


class CustomImportTests(DatasetTestCase):
    def test_text_documents_are_split_on_markers_and_blank_lines(self) -> None:
        text = ("\ufeffFirst document\nstill first.\n<|endoftext|>\nSecond document.\n\n\n\nThird document."
                "\n\nStill third.\r\n  <|endoftext|>  \n\n")
        info = datasets.import_custom("My Notes!", "pretrain", "notes.md", text.encode())
        self.assertEqual(info.id, "custom-my-notes")
        self.assertEqual(list(datasets.iter_documents(info.id)),
                         ["First document\nstill first.", "Second document.", "Third document.\n\nStill third."])
        self.assertEqual(info.description, "Imported from notes.md (3 documents).")

    def test_custom_dataset_lifecycle(self) -> None:
        first = datasets.import_custom("Café Notes", "pretrain", "notes.txt", b"One.\n\n\nTwo.")
        second = datasets.import_custom("Café Notes", "pretrain", "other.txt", b"Three.")
        unnamed = datasets.import_custom("", "pretrain", "\u65e5\u672c.txt", b"Text.")
        self.assertEqual((first.id, second.id, unnamed.id),
                         ("custom-cafe-notes", "custom-cafe-notes-2", "custom-dataset"))
        self.assertEqual((first.name, unnamed.name), ("Café Notes", "\u65e5\u672c"))
        self.assertEqual((first.kind, first.icon, first.license), ("pretrain", "file", "Your own data"))
        self.assertTrue(first.custom and not first.recommended)
        self.assertRegex(first.color, r"^#[0-9a-f]{6}$")
        json.dumps(dataclasses.asdict(first))
        self.assertEqual(datasets.get(first.id), first)
        pretrain_ids = [info.id for info in datasets.catalog("pretrain")]
        self.assertEqual(pretrain_ids[-3:], [first.id, second.id, unnamed.id])
        self.assertNotIn(first.id, [info.id for info in datasets.catalog("finetune")])
        state = datasets.status(first.id)
        self.assertEqual((state["downloaded"], state["documents"], state["downloading"]), (True, 2, False))
        self.assertEqual(datasets.download(first.id), state)
        self.assertEqual(self.web.requests, [])
        with self.assertRaises(ValueError):
            datasets.delete_custom("tinystories")
        datasets.delete_custom(first.id)
        with self.assertRaises(KeyError):
            datasets.get(first.id)
        self.assertEqual([info.id for info in datasets.catalog() if info.custom], [second.id, unnamed.id])

    def test_chat_formats_are_detected(self) -> None:
        rows = [
            {"messages": [system("Be brief."), user("Hi"), assistant("Hello")]},
            {"conversations": [{"from": "human", "value": "2+2?"}, {"from": "gpt", "value": "4"}]},
            {"prompt": "Say yes", "completion": "Yes"},
            {"prompt": "Say no", "response": "No"},
            {"instruction": "Translate", "input": "chat", "output": "cat"},
            {"question": "Capital of France?", "answer": "Paris"},
            {"user": "Ping", "assistant": "Pong", "system": "Network bot."},
            {"Instruction": "Shout", "Output": "HEY"},
            {"text": "no recognisable fields"},
            {"prompt": "Unanswered", "response": ""},
        ]
        data = "\n".join(json.dumps(row) for row in rows).encode() + b"\n{broken json\n"
        info = datasets.import_custom("Mixed", "finetune", "mixed.jsonl", data)
        self.assertEqual([row["messages"] for row in datasets.iter_conversations(info.id)], [
            [system("Be brief."), user("Hi"), assistant("Hello")],
            [user("2+2?"), assistant("4")],
            [user("Say yes"), assistant("Yes")],
            [user("Say no"), assistant("No")],
            [user("Translate\n\nchat"), assistant("cat")],
            [user("Capital of France?"), assistant("Paris")],
            [system("Network bot."), user("Ping"), assistant("Pong")],
            [user("Shout"), assistant("HEY")],
        ])
        self.assertEqual(info.description, "Imported from mixed.jsonl (8 conversations; 3 unusable rows "
                                           "were skipped).")
        self.assertEqual(datasets.status(info.id)["rows"], 8)

    def test_json_containers_and_csv_files(self) -> None:
        cases = [
            ("wrapped.json", json.dumps({"version": 2, "data": [{"question": "Q", "answer": "A"}]}).encode(),
             [[user("Q"), assistant("A")]]),
            ("array.jsonl", json.dumps([{"prompt": "P", "response": "R"}]).encode(),
             [[user("P"), assistant("R")]]),
            ("lines.json", b'{"user": "U1", "assistant": "A1"}\n{"user": "U2", "assistant": "A2"}\n',
             [[user("U1"), assistant("A1")], [user("U2"), assistant("A2")]]),
            ("single.json", json.dumps({"messages": [user("One"), assistant("Only")]}).encode(),
             [[user("One"), assistant("Only")]]),
            ("sheet.csv", '\ufeffInstruction,Input,Output\r\nAdd,1 and 2,3\r\n"Write a\nhaiku",,"Leaves '
                          'fall"\r\n,,\r\n'.encode(),
             [[user("Add\n\n1 and 2"), assistant("3")], [user("Write a\nhaiku"), assistant("Leaves fall")]]),
            ("sharegpt.csv", b'conversations\n"[{""from"": ""human"", ""value"": ""Hi""}, '
                             b'{""from"": ""gpt"", ""value"": ""Hey""}]"\n',
             [[user("Hi"), assistant("Hey")]]),
        ]
        for filename, data, expected in cases:
            with self.subTest(filename=filename):
                info = datasets.import_custom(filename, "finetune", filename, data)
                self.assertEqual([row["messages"] for row in datasets.iter_conversations(info.id)], expected)

    def test_import_errors_are_explained(self) -> None:
        cases = [
            (("X", "poetry", "a.txt", b"hello"), "kind"),
            (("X", "pretrain", "a.pdf", b"hello"), r"\.txt or \.md"),
            (("X", "finetune", "a.txt", b"hello"), r"\.jsonl, \.json or \.csv"),
            (("X", "pretrain", "a.txt", b" \n\t\n"), "empty"),
            (("X", "pretrain", "a.txt", b"abc\x00def"), "binary"),
            (("X", "pretrain", "a.txt", bytes(range(1, 32)) * 10), "binary"),
            (("X", "pretrain", "a.txt", "caf\u00e9".encode("latin-1")), "UTF-8"),
            (("X", "pretrain", "a.txt", b"<|endoftext|>\n<|endoftext|>\n"), "no text"),
            (("X", "finetune", "a.json", b"{not json"), "not valid JSON"),
            (("X", "finetune", "a.json", b'{"a": 1}'), "list of examples"),
            (("X", "finetune", "a.jsonl", b'{"text": "hi"}\n'), "No usable conversations"),
            (("X", "finetune", "a.csv", b"\xff\xfe,\n"), "UTF-8"),
            (("X", "finetune", "a.jsonl", b'{"text": "caf\xe9"}\n'), "UTF-8"),
        ]
        for args, pattern in cases:
            with self.subTest(args=args[1:]), self.assertRaisesRegex(ValueError, pattern):
                datasets.import_custom(*args)
        with mock.patch.object(datasets, "MAX_IMPORT_BYTES", 10):
            with self.assertRaisesRegex(ValueError, "too large"):
                datasets.import_custom("X", "pretrain", "a.txt", b"x" * 11)
        self.assertEqual([info for info in datasets.catalog() if info.custom], [])
        self.assertEqual(self.leftovers(), [])


if __name__ == "__main__":
    unittest.main()
