"""The brain's secrets are bound to its signature (docs/14 section 6, 2026-10-23).

A Developer-ID-signed brain keeps the user's Anthropic key and the sign-in's refresh token in
items it makes itself through Security.framework (`orchestrator/keychain_framework.py`), under
`<service>.v2`, so their access list trusts the brain and not `/usr/bin/security` - which today's
items trust, letting any process of the user read them with no prompt. An item from before moves
over once, read through `/usr/bin/security` (which it trusts, so no prompt) and then deleted.

Every test here works in a THROWAWAY keychain made in its own temporary folder - never the login
keychain (reader S8·R15): the framework calls are handed that keychain, and the one `security`
door is replaced by one that refuses any command not aimed at its file.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import secrets
import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator import device_auth
from orchestrator.keychain_framework import FrameworkStore

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="the macOS keychain")

TOKEN = "tok_" + "a" * 40
TOKEN2 = "tok_" + "b" * 40

_sec = ctypes.CDLL(ctypes.util.find_library("Security")) if sys.platform == "darwin" else None
_cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation")) if sys.platform == "darwin" else None


def _search_list() -> str:
    return subprocess.run(["/usr/bin/security", "list-keychains", "-d", "user"],
                          capture_output=True, text=True).stdout


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """(SecKeychainRef, path) of a new keychain in tmp_path; the `security` door aimed at it."""
    ref = ctypes.c_void_p
    _sec.SecKeychainCreate.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_char_p,
                                       ctypes.c_bool, ref, ctypes.POINTER(ref)]
    _sec.SecKeychainCreate.restype = ctypes.c_int32
    _sec.SecKeychainDelete.argtypes = [ref]
    _cf.CFRelease.argtypes = [ref]
    path = str(tmp_path / "scratch.keychain-db")
    password = secrets.token_hex(16).encode()       # the throwaway keychain's own
    kc = ref()
    before = _search_list()
    assert _sec.SecKeychainCreate(path.encode(), len(password), password, False, None,
                                  ctypes.byref(kc)) == 0

    def door(argv, **kwargs):
        text = " ".join(argv) + " " + kwargs.get("input", "")
        assert path in text, f"a keychain command not aimed at the scratch keychain: {argv}"
        return subprocess.run(argv, capture_output=True, text=True, timeout=20, **kwargs)

    monkeypatch.setattr(device_auth, "_security", door)
    yield kc.value, path
    _sec.SecKeychainDelete(kc)
    _cf.CFRelease(kc)
    assert _search_list() == before, "the user's keychain search list was changed"


def _legacy_add(path: str, service: str, account: str, secret: str) -> None:
    r = subprocess.run(["/usr/bin/security", "add-generic-password", "-s", service, "-a", account,
                        "-w", secret, path], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def _legacy_find(path: str, service: str, account: str) -> int:
    return subprocess.run(["/usr/bin/security", "find-generic-password", "-s", service,
                           "-a", account, path], capture_output=True, text=True).returncode


def _readers(kc: int, service: str, account: str) -> list[str]:
    """Who the item's decrypt ACL trusts, read without asking for the secret (no prompt)."""
    ref, p = ctypes.c_void_p, ctypes.POINTER
    for name, args in {"SecKeychainFindGenericPassword": [ref, ctypes.c_uint32, ctypes.c_char_p,
                                                          ctypes.c_uint32, ctypes.c_char_p,
                                                          p(ctypes.c_uint32), p(ref), p(ref)],
                       "SecKeychainItemCopyAccess": [ref, p(ref)],
                       "SecAccessCopyACLList": [ref, p(ref)],
                       "SecACLCopyContents": [ref, p(ref), p(ref), p(ctypes.c_uint16)],
                       "SecTrustedApplicationCopyData": [ref, p(ref)]}.items():
        getattr(_sec, name).argtypes, getattr(_sec, name).restype = args, ctypes.c_int32
    _sec.SecACLCopyAuthorizations.argtypes, _sec.SecACLCopyAuthorizations.restype = [ref], ref
    _cf.CFArrayGetCount.argtypes, _cf.CFArrayGetCount.restype = [ref], ctypes.c_long
    _cf.CFArrayGetValueAtIndex.argtypes, _cf.CFArrayGetValueAtIndex.restype = [ref, ctypes.c_long], ref
    _cf.CFDataGetLength.argtypes, _cf.CFDataGetLength.restype = [ref], ctypes.c_long
    _cf.CFDataGetBytePtr.argtypes, _cf.CFDataGetBytePtr.restype = [ref], ctypes.c_void_p
    _cf.CFStringGetCString.argtypes = [ref, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
    s, a = service.encode(), account.encode()
    item, access, acls = ref(), ref(), ref()
    assert _sec.SecKeychainFindGenericPassword(kc, len(s), s, len(a), a, None, None,
                                               ctypes.byref(item)) == 0
    _sec.SecKeychainItemCopyAccess(item, ctypes.byref(access))
    _sec.SecAccessCopyACLList(access, ctypes.byref(acls))
    out = []
    for i in range(_cf.CFArrayGetCount(acls)):
        acl = _cf.CFArrayGetValueAtIndex(acls, i)
        auths = _sec.SecACLCopyAuthorizations(acl)
        names = []
        for j in range(_cf.CFArrayGetCount(auths)):
            buf = ctypes.create_string_buffer(128)
            _cf.CFStringGetCString(_cf.CFArrayGetValueAtIndex(auths, j), buf, 128, 0x08000100)
            names.append(buf.value.decode())
        if "ACLAuthorizationDecrypt" not in names and "ACLAuthorizationAny" not in names:
            continue
        apps, desc, prompt = ref(), ref(), ctypes.c_uint16()
        _sec.SecACLCopyContents(acl, ctypes.byref(apps), ctypes.byref(desc), ctypes.byref(prompt))
        if not apps.value:
            out.append("<any application>")
            continue
        for j in range(_cf.CFArrayGetCount(apps)):
            d = ref()
            _sec.SecTrustedApplicationCopyData(_cf.CFArrayGetValueAtIndex(apps, j), ctypes.byref(d))
            out.append(ctypes.string_at(_cf.CFDataGetBytePtr(d), _cf.CFDataGetLength(d))
                       .rstrip(b"\0").decode(errors="replace"))
    return out


def _keychain(kc: int, path: str, account: str = "acct") -> device_auth.Keychain:
    return device_auth.Keychain(account=account,
                                signed=FrameworkStore("com.zammoth.cloud.v2", account, keychain=kc),
                                keychain_path=path)


def test_the_framework_store_reads_writes_modifies_and_deletes(scratch):
    kc, _path = scratch
    store = FrameworkStore("com.zammoth.cloud.v2", "acct", keychain=kc)
    assert store.get() is None
    store.set(b"one")
    assert store.get() == b"one"
    store.set(b"two")
    assert store.get() == b"two"
    store.delete()
    assert store.get() is None
    store.delete()                                    # no item is not an error


def test_an_item_the_framework_makes_trusts_its_maker_and_not_the_security_tool(scratch):
    kc, path = scratch
    FrameworkStore("com.zammoth.cloud.v2", "acct", keychain=kc).set(TOKEN.encode())
    _legacy_add(path, "com.zammoth.cloud", "acct", TOKEN)
    framework, legacy = (_readers(kc, "com.zammoth.cloud.v2", "acct"),
                         _readers(kc, "com.zammoth.cloud", "acct"))
    assert legacy == ["/usr/bin/security"]                  # today: any process may read it
    # bound: one trusted program - the one that made it (here the test's Python; in the
    # shipped app the signed brain) - and neither the security tool nor "any application"
    assert len(framework) == 1, framework
    assert framework[0] not in ("/usr/bin/security", "<any application>")
    # the brain's own reader (what `--keychain-check` prints) says the same as this one
    assert FrameworkStore("com.zammoth.cloud.v2", "acct", keychain=kc).trusted_readers() == framework
    assert FrameworkStore("com.zammoth.cloud", "acct", keychain=kc).trusted_readers() == legacy


def test_an_item_from_before_moves_over_once_and_the_old_one_goes(scratch):
    kc, path = scratch
    _legacy_add(path, "com.zammoth.cloud", "acct", TOKEN)
    keychain = _keychain(kc, path)
    assert keychain.get() == TOKEN
    assert FrameworkStore("com.zammoth.cloud.v2", "acct", keychain=kc).get() == TOKEN.encode()
    assert _legacy_find(path, "com.zammoth.cloud", "acct") == 44        # gone
    assert keychain.get() == TOKEN                                       # and read from the new one


def test_a_write_goes_to_the_bound_item_and_clears_an_old_copy(scratch):
    kc, path = scratch
    _legacy_add(path, "com.zammoth.cloud", "acct", TOKEN)
    keychain = _keychain(kc, path)
    keychain.set(TOKEN2)
    assert FrameworkStore("com.zammoth.cloud.v2", "acct", keychain=kc).get() == TOKEN2.encode()
    assert _legacy_find(path, "com.zammoth.cloud", "acct") == 44
    keychain.delete()
    assert keychain.get() is None


def test_a_malformed_secret_never_reaches_the_keychain(scratch):
    kc, path = scratch
    with pytest.raises(device_auth.DeviceAuthError):
        _keychain(kc, path).set("not a token; rm -rf /")
    assert FrameworkStore("com.zammoth.cloud.v2", "acct", keychain=kc).get() is None


def test_a_source_checkout_keeps_the_security_tool(monkeypatch):
    # not frozen: never bound (its program is Python; a dev brain's identity also changes with
    # every build). Asked fresh here: conftest pins every test brain to "not signed".
    from orchestrator import llm_backend
    monkeypatch.setattr(device_auth, "_signed_team", False)
    monkeypatch.setattr(llm_backend, "is_frozen", lambda: False)
    assert device_auth.signed_brain_team() is None
    assert device_auth.Keychain(account="acct")._signed_store() is None


def test_a_frozen_brain_is_bound_only_when_its_signature_names_a_team(monkeypatch):
    from orchestrator import llm_backend
    monkeypatch.setattr(llm_backend, "is_frozen", lambda: True)
    for stderr, team in (("Identifier=mixagent_daw_brain\nTeamIdentifier=K7KT9G3TZF\n", "K7KT9G3TZF"),
                         ("Identifier=mixagent_daw_brain\nTeamIdentifier=not set\n", None),
                         ("Signature=adhoc\n", None)):
        monkeypatch.setattr(device_auth, "_signed_team", False)
        monkeypatch.setattr(device_auth.subprocess, "run",
                            lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "", stderr))
        assert device_auth.signed_brain_team() == team


def test_no_test_brain_is_a_signed_one():
    # the conftest guard: the framework path never reaches the login keychain from a test
    assert device_auth._signed_team is None
    assert device_auth.Keychain(account="acct")._signed_store() is None
