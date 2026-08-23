"""Tests for G-VITATTN.

A gate that cannot fail is worse than no gate, because it produces a reassuring log line.
This gate exists *because* a config setting was accepted, logged, and ignored (job 3167568),
so every one of its failure branches is shown here to actually fire on a planted violation --
including the exact unpatched-vLLM behaviour it was written to catch.

vLLM is not installed locally, so the module is faked. That is appropriate: what is under
test is our gate's decision logic, not vLLM.
"""

from __future__ import annotations

import enum
import sys
import tempfile
import types
import unittest
from pathlib import Path

import container_gate as G


class FakeBackend(enum.Enum):
    TORCH_SDPA = "TORCH_SDPA"
    FLASH_ATTN = "FLASH_ATTN"
    ROCM_AITER_FA = "ROCM_AITER_FA"


def install_fake_vllm(returns, module_body: bytes = b"# fake vllm layer\n"):
    """Put a fake vllm.attention.layer on sys.modules whose function returns `returns`.

    The module gets a REAL file on disk so the gate's sha256 identity check operates on
    something genuine rather than a stub.
    """
    tmp = Path(tempfile.mkdtemp(prefix="ca21_fakevllm_")) / "layer.py"
    tmp.write_bytes(module_body)

    registry = types.ModuleType("vllm.attention.backends.registry")
    registry.AttentionBackendEnum = FakeBackend

    layer = types.ModuleType("vllm.attention.layer")
    layer.__file__ = str(tmp)
    layer.maybe_get_vit_flash_attn_backend = lambda *a, **k: returns

    mods = {
        "vllm": types.ModuleType("vllm"),
        "vllm.attention": types.ModuleType("vllm.attention"),
        "vllm.attention.backends": types.ModuleType("vllm.attention.backends"),
        "vllm.attention.backends.registry": registry,
        "vllm.attention.layer": layer,
    }
    sys.modules.update(mods)
    return tmp, list(mods)


class GateBase(unittest.TestCase):
    def tearDown(self):
        for name in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
            del sys.modules[name]


class TestBehaviourCheck(GateBase):
    def test_passes_when_override_is_honoured(self):
        install_fake_vllm((FakeBackend.TORCH_SDPA, None))
        out = G.assert_vit_attn_patch(verbose=False)
        self.assertEqual(out["backend"], str(FakeBackend.TORCH_SDPA))
        self.assertFalse(out["identity_checked"])

    def test_fires_on_the_real_unpatched_vllm_behaviour(self):
        """THE case this gate exists for: override silently reverted to FLASH_ATTN.

        This is what job 3167568 did -- and it is why that job failed identically to the
        job before the fix, with a log line claiming TORCH_SDPA throughout.
        """
        install_fake_vllm((FakeBackend.FLASH_ATTN, lambda: None))
        with self.assertRaises(AssertionError) as cm:
            G.assert_vit_attn_patch(verbose=False)
        msg = str(cm.exception)
        self.assertIn("G-VITATTN FAILED", msg)
        self.assertIn("FLASH_ATTN", msg)
        # It must say what to go and look at, not merely that something is wrong.
        self.assertIn("head_dim=80", msg)
        self.assertIn(G.CONTAINER_LAYER_PATH, msg)

    def test_fires_when_sdpa_still_carries_a_flash_attn_function(self):
        """Backend right, binding wrong -- an inconsistent state we should never proceed on."""
        install_fake_vllm((FakeBackend.TORCH_SDPA, lambda: None))
        with self.assertRaises(AssertionError) as cm:
            G.assert_vit_attn_patch(verbose=False)
        self.assertIn("still bound", str(cm.exception))


class TestIdentityCheck(GateBase):
    def test_identity_enforced_only_when_a_hash_is_supplied(self):
        install_fake_vllm((FakeBackend.TORCH_SDPA, None))
        G.assert_vit_attn_patch(expect_sha256=None, verbose=False)  # not enforced

    def test_fires_when_the_mounted_file_is_not_ours(self):
        """Behaviour correct but identity wrong: needs a human, not a silent pass."""
        install_fake_vllm((FakeBackend.TORCH_SDPA, None))
        with self.assertRaises(AssertionError) as cm:
            G.assert_vit_attn_patch(expect_sha256="0" * 64, verbose=False)
        self.assertIn("identity", str(cm.exception))

    def test_passes_when_hash_matches(self):
        body = b"# exact bytes\n"
        path, _ = install_fake_vllm((FakeBackend.TORCH_SDPA, None), module_body=body)
        out = G.assert_vit_attn_patch(expect_sha256=G.sha256_file(path), verbose=False)
        self.assertTrue(out["identity_checked"])


class TestPinnedOriginal(unittest.TestCase):
    def test_pin_matches_the_committed_pin_file(self):
        """The constant and ORIGINAL.sha256 must not drift apart."""
        pin = (Path(__file__).resolve().parent.parent
               / "patches" / "vllm_0_11_2" / "ORIGINAL.sha256")
        self.assertTrue(pin.exists(), f"missing {pin}")
        self.assertEqual(pin.read_text().split()[0], G.ORIGINAL_SHA256)

    def test_patched_file_differs_from_the_pinned_original(self):
        """Proves the patch is a real edit and the pin names the pre-patch bytes."""
        d = Path(__file__).resolve().parent.parent / "patches" / "vllm_0_11_2"
        self.assertNotEqual(G.sha256_file(d / "layer.py"), G.ORIGINAL_SHA256)

    def test_patched_file_hash_matches_its_own_pin(self):
        d = Path(__file__).resolve().parent.parent / "patches" / "vllm_0_11_2"
        declared = (d / "PATCHED.sha256").read_text().split()[0]
        self.assertEqual(G.sha256_file(d / "layer.py"), declared)

    def test_format_check_pins_the_same_patched_hash(self):
        """format_check.py hardcodes the hash; it must not drift from the file.

        Without this, editing layer.py would leave format_check asserting a stale hash and
        G-VITATTN would fail on a correct patch -- or worse, pass on a stale one.
        """
        import re
        d = Path(__file__).resolve().parent.parent / "patches" / "vllm_0_11_2"
        src = (Path(__file__).resolve().parent / "format_check.py").read_text()
        m = re.search(r'PATCHED_LAYER_SHA256\s*=\s*"([0-9a-f]{64})"', src)
        self.assertIsNotNone(m, "PATCHED_LAYER_SHA256 not found in format_check.py")
        self.assertEqual(m.group(1), G.sha256_file(d / "layer.py"))

    def test_patched_file_contains_the_guard_and_only_that_change(self):
        """Substantive check: the ROCm guard now appears in the CUDA branch too."""
        d = Path(__file__).resolve().parent.parent / "patches" / "vllm_0_11_2"
        src = (d / "layer.py").read_text()
        self.assertEqual(src.count("attn_backend_override is None"), 2,
                         "expected the guard in BOTH the ROCm and CUDA branches")
        self.assertIn("CA21 PATCH", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
