"""The login keychain through Security.framework, for a brain signed with a Developer ID (2026-10-23).

`device_auth.Keychain` stored the user's Anthropic key and the sign-in's refresh token through
`/usr/bin/security`. An item made that way trusts `/usr/bin/security`, so any process running as
the user could read the secret with `security find-generic-password -w` and no prompt (docs/14
section 6, reader S8·R10). An item made by a program through the framework trusts THAT program -
by its code requirement, which for the signed brain is its identifier and the team
(`identifier "mixagent_daw_brain" ... certificate leaf[subject.OU] = K7KT9G3TZF`), so the next
signed build reads it silently and anything else, `/usr/bin/security` included, has to ask the
user for their login password. Measured on macOS 27 with a throwaway keychain before this was
written: the framework item's decrypt ACL names only its creator; the `security` item's names
`/usr/bin/security`.

ctypes over the generic-password calls (`SecKeychainAddGenericPassword`,
`SecKeychainFindGenericPassword`, `SecKeychainItemModifyAttributesAndData`,
`SecKeychainItemDelete`): deprecated, still shipped, stdlib only. The data-protection keychain
would need a keychain-access-group entitlement and a provisioning profile.

Every call runs on a thread of its own and is waited for at most `TIMEOUT_S`: a locked keychain,
or an item this program does not own, puts up a macOS prompt, and the call blocks until it is
answered. A call that outlives the bound is read as "the keychain did not answer" (the thread is
left to finish; it holds nothing of ours).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import threading
from typing import Callable, TypeVar

#: Status codes (Security/SecBase.h).
OK = 0
NOT_FOUND = -25300          # errSecItemNotFound
DUPLICATE = -25299          # errSecDuplicateItem

TIMEOUT_S = 20.0

_T = TypeVar("_T")
_lib: "tuple[ctypes.CDLL, ctypes.CDLL] | None" = None


class FrameworkError(RuntimeError):
    """A call the framework refused (`status`), or one that did not answer (`status` None)."""

    def __init__(self, what: str, status: "int | None") -> None:
        super().__init__(f"{what}: " + ("no answer" if status is None else f"status {status}"))
        self.status = status


def _libs() -> "tuple[ctypes.CDLL, ctypes.CDLL]":
    global _lib
    if _lib is None:
        sec = ctypes.CDLL(ctypes.util.find_library("Security"))
        cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
        u32, ref, p = ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER
        for name, args in {
            "SecKeychainAddGenericPassword": [ref, u32, ctypes.c_char_p, u32, ctypes.c_char_p,
                                              u32, ctypes.c_char_p, p(ref)],
            "SecKeychainFindGenericPassword": [ref, u32, ctypes.c_char_p, u32, ctypes.c_char_p,
                                               p(u32), p(ctypes.c_void_p), p(ref)],
            "SecKeychainItemModifyAttributesAndData": [ref, ref, u32, ctypes.c_char_p],
            "SecKeychainItemFreeContent": [ref, ctypes.c_void_p],
            "SecKeychainItemDelete": [ref],
        }.items():
            fn = getattr(sec, name)
            fn.argtypes, fn.restype = args, ctypes.c_int32
        cf.CFRelease.argtypes = [ref]
        _lib = (sec, cf)
    return _lib


def _bounded(what: str, call: "Callable[[], _T]", timeout_s: float) -> _T:
    box: dict = {}

    def run() -> None:
        try:
            box["value"] = call()
        except BaseException as exc:          # noqa: BLE001 - carried to the caller
            box["error"] = exc

    worker = threading.Thread(target=run, name=f"keychain {what}", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise FrameworkError(what, None)
    if "error" in box:
        raise box["error"]
    return box["value"]


class FrameworkStore:
    """One generic password (service, account) in a keychain - the default search list and the
    login keychain when `keychain` is None, or the keychain a `SecKeychainRef` names (tests)."""

    def __init__(self, service: str, account: str, *, keychain: "int | None" = None,
                 timeout_s: float = TIMEOUT_S) -> None:
        self.service = service.encode()
        self.account = account.encode()
        self.keychain = keychain
        self.timeout_s = timeout_s

    def _find(self, want_data: bool) -> "tuple[int, bytes | None, int | None]":
        sec, _cf = _libs()
        n, data, item = ctypes.c_uint32(), ctypes.c_void_p(), ctypes.c_void_p()
        status = sec.SecKeychainFindGenericPassword(
            self.keychain, len(self.service), self.service, len(self.account), self.account,
            ctypes.byref(n) if want_data else None, ctypes.byref(data) if want_data else None,
            ctypes.byref(item))
        value = None
        if status == OK and want_data:
            value = ctypes.string_at(data, n.value)
            sec.SecKeychainItemFreeContent(None, data)
        return status, value, item.value

    def get(self) -> "bytes | None":
        """The secret, or None when there is no such item. Raises FrameworkError otherwise."""
        def call() -> "bytes | None":
            status, value, item = self._find(True)
            if item:
                _libs()[1].CFRelease(item)
            if status == NOT_FOUND:
                return None
            if status != OK:
                raise FrameworkError("read", status)
            return value
        return _bounded("read", call, self.timeout_s)

    def set(self, secret: bytes) -> None:
        def call() -> None:
            sec, cf = _libs()
            status, _value, item = self._find(False)
            try:
                if status == OK:
                    status = sec.SecKeychainItemModifyAttributesAndData(item, None, len(secret), secret)
                elif status == NOT_FOUND:
                    status = sec.SecKeychainAddGenericPassword(
                        self.keychain, len(self.service), self.service,
                        len(self.account), self.account, len(secret), secret, None)
            finally:
                if item:
                    cf.CFRelease(item)
            if status != OK:
                raise FrameworkError("write", status)
        _bounded("write", call, self.timeout_s)

    def trusted_readers(self) -> list[str]:
        """Who the item's decrypt ACL trusts - each program's path, or "<any application>" -
        read from the item's access object, which never asks for the secret (no prompt)."""
        def call() -> list[str]:
            sec, cf = _libs()
            ref, p = ctypes.c_void_p, ctypes.POINTER
            for name, args in {"SecKeychainItemCopyAccess": [ref, p(ref)],
                               "SecAccessCopyACLList": [ref, p(ref)],
                               "SecACLCopyContents": [ref, p(ref), p(ref), p(ctypes.c_uint16)],
                               "SecTrustedApplicationCopyData": [ref, p(ref)]}.items():
                fn = getattr(sec, name)
                fn.argtypes, fn.restype = args, ctypes.c_int32
            sec.SecACLCopyAuthorizations.argtypes = [ref]
            sec.SecACLCopyAuthorizations.restype = ref
            cf.CFArrayGetCount.argtypes, cf.CFArrayGetCount.restype = [ref], ctypes.c_long
            cf.CFArrayGetValueAtIndex.argtypes = [ref, ctypes.c_long]
            cf.CFArrayGetValueAtIndex.restype = ref
            cf.CFDataGetLength.argtypes, cf.CFDataGetLength.restype = [ref], ctypes.c_long
            cf.CFDataGetBytePtr.argtypes, cf.CFDataGetBytePtr.restype = [ref], ctypes.c_void_p
            cf.CFStringGetCString.argtypes = [ref, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
            status, _value, item = self._find(False)
            if status != OK:
                raise FrameworkError("read access", status)
            access, acls, out = ref(), ref(), []
            sec.SecKeychainItemCopyAccess(item, ctypes.byref(access))
            sec.SecAccessCopyACLList(access, ctypes.byref(acls))
            for i in range(cf.CFArrayGetCount(acls)):
                acl = cf.CFArrayGetValueAtIndex(acls, i)
                auths = sec.SecACLCopyAuthorizations(acl)
                names = []
                for j in range(cf.CFArrayGetCount(auths)):
                    buf = ctypes.create_string_buffer(128)
                    cf.CFStringGetCString(cf.CFArrayGetValueAtIndex(auths, j), buf, 128, 0x08000100)
                    names.append(buf.value.decode())
                if not {"ACLAuthorizationDecrypt", "ACLAuthorizationAny"} & set(names):
                    continue
                apps, desc, prompt = ref(), ref(), ctypes.c_uint16()
                sec.SecACLCopyContents(acl, ctypes.byref(apps), ctypes.byref(desc), ctypes.byref(prompt))
                if not apps.value:
                    out.append("<any application>")
                    continue
                for j in range(cf.CFArrayGetCount(apps)):
                    data = ref()
                    sec.SecTrustedApplicationCopyData(cf.CFArrayGetValueAtIndex(apps, j),
                                                      ctypes.byref(data))
                    out.append(ctypes.string_at(cf.CFDataGetBytePtr(data), cf.CFDataGetLength(data))
                               .rstrip(b"\0").decode(errors="replace"))
            cf.CFRelease(item)
            return out
        return _bounded("read access", call, self.timeout_s)

    def delete(self) -> None:
        """Remove the item; no item is not an error."""
        def call() -> None:
            sec, cf = _libs()
            status, _value, item = self._find(False)
            try:
                if status == OK:
                    status = sec.SecKeychainItemDelete(item)
            finally:
                if item:
                    cf.CFRelease(item)
            if status not in (OK, NOT_FOUND):
                raise FrameworkError("delete", status)
        _bounded("delete", call, self.timeout_s)


#: The dummy item `--keychain-check` makes - never a real secret's service.
CHECK_SERVICE = "com.zammoth.keychain-check.v2"
CHECK_ACCOUNT = "check"


def self_check(argv: list[str]) -> int:
    """`mixagent_daw_brain --keychain-check [--keep | --read]`: what THIS program's keychain items
    are bound to, proven on the login keychain with a dummy item (docs/14 section 6).

    Default: write a random dummy value, read it back, report who the item trusts, delete it.
    `--keep` leaves the item (to read it from another build); `--read` only reads an item a
    `--keep` left and then deletes it. Prints one JSON line; never prints the value."""
    import json
    import secrets
    import sys

    from .device_auth import signed_brain_team

    store = FrameworkStore(CHECK_SERVICE, CHECK_ACCOUNT)
    report: dict = {"executable": sys.executable, "team": signed_brain_team()}
    try:
        if "--read" in argv:
            got = store.get()
            report["found"] = got is not None
            if got is not None:
                report["readers"] = store.trusted_readers()
            store.delete()
        else:
            value = secrets.token_hex(24).encode()
            store.set(value)
            report["read_back"] = store.get() == value
            report["readers"] = store.trusted_readers()
            if "--keep" not in argv:
                store.delete()
                report["deleted"] = store.get() is None
    except FrameworkError as exc:
        report["error"] = str(exc)
    print(json.dumps(report))
    ok = "error" not in report and report.get("read_back", report.get("found", False))
    return 0 if ok else 1
