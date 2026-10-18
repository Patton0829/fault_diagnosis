"""Synthetic, offline contracts for adjacent chunk context expansion.

No test reads the real corpus, index, cache, or .env. CLI tests replace the ready
reader and forbid model requests and Milvus/HTTP SDK imports.
"""

import builtins
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import importlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

module = importlib.import_module("qa_platform.retrieval.context_expansion")


def records_for(texts, starts=None):
    """Build body-only records in one synthetic document and one section."""
    records = []
    offset = 0
    for index, text in enumerate(texts):
        start = offset if starts is None else starts[index]
        identity = {
            "chunk_id": f"chunk-{index}",
            "parent_id": "parent-a",
            "kb_id": "synthetic-kubernetes",
            "snapshot_commit": "a" * 40,
            "review_status": "upstream_unreviewed",
        }
        metadata = {
            **identity,
            "chunk_index": index,
            "chunk_count": len(texts),
            "cleaned_char_start": start,
            "cleaned_char_end": start + len(text),
            "sections": [["调试", "DNS"]],
            "offset_scope": module.OFFSET_SCOPE,
            "heading_context_ranges": [],
            "source": "synthetic-kubernetes",
            "source_url": "https://example.invalid/dns",
            "cleaned_sha256": "b" * 64,
        }
        records.append({**identity, "text": text, "metadata": metadata})
        offset = start + len(text)
    return records


def hits_for(records, indexes, score=0.75):
    return [
        {
            "chunk_id": records[index]["chunk_id"],
            "record": records[index],
            "score": score,
        }
        for index in indexes
    ]


class ContextExpansionTests(unittest.TestCase):
    def test_adjacent_command_and_output_follow_source_order(self):
        records = records_for(
            ["$ nslookup service\n", "Server: 10.0.0.10\n", "Name: service\n"]
        )
        result = module.expand_context(hits_for(records, [1]), list(reversed(records)))

        self.assertEqual(result["selected_ids"], ["chunk-0", "chunk-1", "chunk-2"])
        self.assertEqual(result["added_ids"], ["chunk-0", "chunk-2"])
        self.assertEqual(len(result["passages"]), 1)
        passage = result["passages"][0]
        self.assertEqual(passage["text"], "".join(row["text"] for row in records))
        self.assertEqual((passage["start"], passage["end"]), (0, len(passage["text"])))
        self.assertEqual(passage["seed_ranks"], [1])
        self.assertFalse(any("score" in event for event in result["events"]))
        self.assertNotIn("score", passage)

    def test_two_seeds_reuse_their_shared_neighbor_once(self):
        records = records_for(["a", "b", "c", "d", "e"])
        result = module.expand_context(hits_for(records, [1, 3]), records)

        self.assertEqual(
            result["selected_ids"], [f"chunk-{index}" for index in range(5)]
        )
        self.assertEqual(result["added_ids"], ["chunk-0", "chunk-2", "chunk-4"])
        self.assertEqual(result["passages"][0]["text"], "abcde")
        self.assertEqual(result["passages"][0]["seed_ranks"], [1, 2])
        shared = [
            event for event in result["events"] if event["neighbor_id"] == "chunk-2"
        ]
        self.assertEqual([event["status"] for event in shared], ["added", "reused"])

    def test_matching_overlap_is_deduplicated_before_budgeting(self):
        records = records_for(["abcdef", "defghi", "ghiJKL"], starts=[0, 3, 6])
        result = module.expand_context(
            hits_for(records, [0, 1]), records, max_context_chars=12
        )

        self.assertEqual(result["baseline_chars"], 12)
        self.assertEqual(result["baseline_unique_chars"], 9)
        self.assertEqual(result["expanded_chars"], 12)
        self.assertEqual(result["passages"][0]["text"], "abcdefghiJKL")
        self.assertEqual(result["added_ids"], ["chunk-2"])

    def test_conflicting_overlap_is_rejected(self):
        records = records_for(["abcdef", "dXfghi"], starts=[0, 3])
        with self.assertRaisesRegex(ValueError, "重叠区文字不一致"):
            module.expand_context(hits_for(records, [0]), records)

    def test_gap_rejects_neighbor_and_stops_that_direction(self):
        records = records_for(["a", "b", "c", "d"], starts=[0, 1, 5, 6])
        result = module.expand_context(hits_for(records, [2]), records, window=3)

        self.assertEqual(result["selected_ids"], ["chunk-2", "chunk-3"])
        self.assertEqual(result["passages"][0]["text"], "cd")
        before = [event for event in result["events"] if event["direction"] == -1]
        self.assertEqual(len(before), 1)
        self.assertEqual(before[0]["reason"], "non_contiguous_range")
        self.assertEqual(before[0]["neighbor_id"], "chunk-1")

    def test_seed_passages_across_gap_stay_separate_and_keep_rank(self):
        records = records_for(["first", "second"], starts=[0, 20])
        result = module.expand_context(hits_for(records, [1, 0]), records, window=0)

        self.assertEqual([p["text"] for p in result["passages"]], ["second", "first"])
        self.assertEqual([p["seed_ranks"] for p in result["passages"]], [[1], [2]])
        self.assertEqual(result["expanded_chars"], 11)

    def test_section_parent_and_source_versions_are_boundaries(self):
        changes = {
            "sections": [["调试", "网络"]],
            "parent_id": "parent-b",
            "snapshot_commit": "c" * 40,
            "cleaned_sha256": "d" * 64,
            "source_url": "https://example.invalid/other",
            "corpus_relative_path": "other.md",
            "strategy": "other-splitter",
            "target_chars": 100,
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                records = records_for(["a", "b", "c"])
                records[0]["metadata"][field] = value
                if field in records[0]:
                    records[0][field] = value
                result = module.expand_context(hits_for(records, [1]), records)
                self.assertEqual(result["selected_ids"], ["chunk-1", "chunk-2"])
                self.assertEqual(result["events"][0]["status"], "skipped")

    def test_budget_keeps_seeds_and_allocates_by_distance_rank_then_side(self):
        records = records_for(list("abcdefg"))
        hits = hits_for(records, [2, 5])
        result = module.expand_context(hits, records, window=2, max_context_chars=4)

        self.assertEqual(result["seed_ids"], ["chunk-2", "chunk-5"])
        self.assertEqual(result["added_ids"], ["chunk-1", "chunk-3"])
        self.assertEqual(result["hits"], hits)
        self.assertEqual(result["expanded_chars"], 4)
        self.assertEqual(
            [
                (event["distance"], event["seed_rank"], event["direction"])
                for event in result["events"][:4]
            ],
            [(1, 1, -1), (1, 1, 1), (1, 2, -1), (1, 2, 1)],
        )
        self.assertTrue(
            all(
                event["distance"] == 1
                for event in result["events"]
                if event["seed_rank"] == 2
            )
        )
        self.assertEqual(result["events"][2]["reason"], "budget_exceeded")
        self.assertEqual(result["passages"][0]["text"], "bcd")
        self.assertEqual(result["passages"][1]["text"], "f")

    def test_budget_smaller_than_original_hits_is_rejected_without_truncation(self):
        records = records_for(["original evidence"])
        with self.assertRaisesRegex(ValueError, "不丢弃原命中、不截断"):
            module.expand_context(hits_for(records, [0]), records, max_context_chars=3)

    def test_window_zero_keeps_manual_seed_unscored_without_adding_neighbors(self):
        records = records_for(["before", "seed", "after"])
        result = module.expand_context(
            hits_for(records, [1], score=None), records, window=0
        )

        self.assertEqual(result["selected_ids"], ["chunk-1"])
        self.assertEqual(result["added_ids"], [])
        self.assertEqual(result["events"], [])
        self.assertIsNone(result["hits"][0]["score"])
        self.assertEqual(result["passages"][0]["text"], "seed")

    def test_missing_metadata_and_heading_prefix_are_rejected(self):
        changes = [
            ("cleaned_sha256", None),
            ("heading_context_ranges", [[0, 2]]),
            ("offset_scope", "raw_file_bytes"),
            ("sections", [["DNS"], ["网络"]]),
        ]
        for field, value in changes:
            with self.subTest(field=field):
                records = records_for(["seed"])
                if value is None:
                    del records[0]["metadata"][field]
                else:
                    records[0]["metadata"][field] = value
                with self.assertRaises(ValueError):
                    module.expand_context(hits_for(records, [0]), records)

    def test_input_and_returned_nested_metadata_are_independent(self):
        records = records_for(["before", "seed", "after"])
        hits = hits_for(records, [1])
        original_records, original_hits = deepcopy(records), deepcopy(hits)
        result = module.expand_context(hits, records)

        self.assertEqual(records, original_records)
        self.assertEqual(hits, original_hits)
        self.assertIsNot(result["hits"][0]["record"], records[1])
        result["hits"][0]["record"]["metadata"]["sections"][0].append("changed")
        result["passages"][0]["metadata"]["sections"][0].append("changed")
        self.assertEqual(records, original_records)
        self.assertEqual(hits, original_hits)

    def test_retrieve_once_encodes_and_searches_each_route_exactly_once(self):
        search = importlib.import_module("qa_platform.storage.kubernetes_search")
        records = records_for(["seed"])
        hits = hits_for(records, [0])
        client = Mock()
        client_factory = Mock(return_value=client)
        session_factory = object()
        query = object()
        with tempfile.TemporaryDirectory(prefix="context-retrieve-tests-") as directory:
            database = Path(directory) / search.build.DATABASE
            database.mkdir()
            check = {
                "index_dir": Path(directory),
                "plan": {},
                "bundle": {"vectors": [[1.0, 0.0]]},
                "vectors": [[1.0, 0.0]],
                "records": records,
                "top_k": 1,
                "filter": 'parent_id == "parent-a"',
            }
            with ExitStack() as stack:
                read = stack.enter_context(
                    patch.object(search, "read_index", return_value=[[1.0, 0.0]])
                )
                encode = stack.enter_context(
                    patch.object(
                        search,
                        "encode_query",
                        return_value=(
                            query,
                            {"embedding_requests": 1, "http_requests": 3},
                        ),
                    )
                )
                rank = stack.enter_context(
                    patch.object(search, "rank_numpy", return_value=hits)
                )
                retrieve = stack.enter_context(
                    patch.object(search, "search_milvus", return_value=hits)
                )
                compare = stack.enter_context(
                    patch.object(
                        search,
                        "validate_top_k",
                        return_value={"max_score_difference": 0.0},
                    )
                )
                stack.enter_context(redirect_stdout(io.StringIO()))
                actual = module.retrieve_once(
                    check, 17, session_factory, client_factory
                )

            self.assertEqual(actual, hits)
            client_factory.assert_called_once_with(uri=str(database), timeout=30)
            read.assert_called_once_with(client, check["plan"])
            encode.assert_called_once_with(check, 17, session_factory)
            rank.assert_called_once_with(records, check["vectors"], query, 1)
            retrieve.assert_called_once_with(client, records, query, 1, check["filter"])
            compare.assert_called_once_with(hits, records, check["vectors"], query, 1)
            client.close.assert_called_once_with()

    def test_retrieve_once_closes_mismatching_index_before_model_encoding(self):
        search = importlib.import_module("qa_platform.storage.kubernetes_search")
        client = Mock()
        with tempfile.TemporaryDirectory(prefix="context-retrieve-tests-") as directory:
            (Path(directory) / search.build.DATABASE).mkdir()
            check = {
                "index_dir": Path(directory),
                "plan": {},
                "bundle": {"vectors": [[1.0, 0.0]]},
            }
            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(search, "read_index", return_value=[[0.0, 1.0]])
                )
                encode = stack.enter_context(patch.object(search, "encode_query"))
                retrieve = stack.enter_context(patch.object(search, "search_milvus"))
                with self.assertRaisesRegex(ValueError, "缓存向量不一致"):
                    module.retrieve_once(
                        check, client_factory=Mock(return_value=client)
                    )
            encode.assert_not_called()
            retrieve.assert_not_called()
            client.close.assert_called_once_with()

    def run_offline_cli(self, arguments):
        search = importlib.import_module("qa_platform.storage.kubernetes_search")
        records = records_for(["before\n", "seed\n", "after\n"])
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in {
                "pymilvus",
                "milvus_lite",
                "requests",
                "dotenv",
            }:
                raise AssertionError(f"offline CLI imported {name}")
            return original_import(name, *args, **kwargs)

        output = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    search,
                    "prepare_search",
                    return_value={
                        "records": records,
                        "query": "synthetic DNS question",
                        "top_k": 3,
                        "filter": 'parent_id == "parent-a"',
                    },
                )
            )
            for name in ("run_search", "encode_query", "read_index", "http_json"):
                stack.enter_context(
                    patch.object(
                        search, name, side_effect=AssertionError(f"called {name}")
                    )
                )
            stack.enter_context(
                patch.object(
                    module,
                    "retrieve_once",
                    side_effect=AssertionError("unexpected retrieval"),
                )
            )
            stack.enter_context(
                patch("builtins.__import__", side_effect=guarded_import)
            )
            stack.enter_context(
                patch.object(sys, "argv", ["context_expansion", *arguments])
            )
            stack.enter_context(redirect_stdout(output))
            self.assertEqual(module.main(), 0)
        return output.getvalue()

    def test_default_cli_only_prints_zero_http_plan(self):
        output = self.run_offline_cli([])
        self.assertIn("实际HTTP=0", output)
        self.assertIn("未导入Milvus SDK", output)
        self.assertNotIn("[原块 1]", output)

    def test_manual_cli_run_prints_unscored_context_without_http_or_sdk(self):
        output = self.run_offline_cli(["--run", "--seed-chunk", "chunk-1"])
        self.assertIn("HTTP=0", output)
        self.assertIn("未评分（手工指定，不是本次检索结果）", output)
        self.assertIn("before\nseed\nafter\n", output)
        self.assertIn("补充邻块", output)
        self.assertNotIn("COSINE=", output)


if __name__ == "__main__":
    unittest.main()
