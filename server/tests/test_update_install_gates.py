"""The update's install asks its two questions AGAIN when the download lands (reader S7·R6,
2026-10-05, batch M15).

`offerUpdate` asked "an export is rendering" and "save your changes?" once, then a zip of some
hundred megabytes downloaded, then `installAndRelaunch` quit the app. Whatever the app did on its
own during the download - an agent turn still applying moves, an export it started - was dropped
by that quit with no prompt and, being a clean exit, no recovery snapshot.

`MixAgentDawApplication` is the GUI application object: no console leg can construct it. So the
shape is pinned in the source - there is ONE call of `installAndRelaunch`, it sits inside
`finishInstall`, behind both gates in order, and the download's callback reaches it only through
that function.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = (ROOT / "daw" / "Source" / "App.cpp").read_text(encoding="utf-8")
UPDATER = (ROOT / "daw" / "Source" / "Updater.h").read_text(encoding="utf-8")


def _code(text: str) -> str:
    """The source with its // comments taken out (a comment may name the function)."""
    return re.sub(r"//[^\n]*", "", text)


def _body(text: str, signature: str) -> str:
    """The braces of the function whose definition starts with `signature`."""
    start = text.index(signature)
    depth, i = 0, text.index("{", start)
    for j in range(i, len(text)):
        depth += text[j] == "{"
        depth -= text[j] == "}"
        if depth == 0:
            return text[i : j + 1]
    raise AssertionError(f"unbalanced braces after {signature}")


def test_the_install_is_reached_only_through_both_gates() -> None:
    app = _code(APP)
    assert app.count("installAndRelaunch (") == 1, "one door to the helper: finishInstall"
    finish = _body(app, "void finishInstall (")
    export_at = finish.index('askAboutRunningExport ("Install"')
    save_at = finish.index('confirmSaveThen ("installing the update"')
    install_at = finish.index("installAndRelaunch (")
    assert export_at < save_at < install_at, "the export question, then the save question, then the install"


def test_the_download_callback_goes_through_finish_install() -> None:
    begin = _body(_code(APP), "void beginInstall (")
    assert "finishInstall (" in begin
    assert "installAndRelaunch" not in begin, "the download's callback must not quit the app itself"


def test_the_first_ask_is_still_made_before_the_download() -> None:
    offer = _body(_code(APP), "void offerUpdate (")
    assert offer.index('askAboutRunningExport ("Install"') < offer.index("beginInstall (")


def test_the_app_does_not_quit_when_the_helper_did_not_start() -> None:
    install = _body(_code(UPDATER), "inline bool installAndRelaunch (")
    assert install.index("helper.start (") < install.index("->quit()")
    assert re.search(r"if \(! helper\.start \(.*?\)\)\s*return false;", install, re.S), \
        "a helper that could not be started leaves the app running"


def test_whether_this_copy_can_be_replaced_is_asked_before_the_download() -> None:
    """Reader S7.R4: an app an administrator installed, or one run from a read-only place,
    downloaded, quit, failed in the helper and offered the same update again."""
    begin = _body(_code(APP), "void beginInstall (")
    assert begin.index("zupdate::installBlocker (") < begin.index("zupdate::downloadAsync (")


def test_the_download_is_hashed_before_its_callback() -> None:
    download = _body(_code(UPDATER), "inline void downloadAsync (")
    assert download.index("fetchFile (") < download.index("downloadProblem (") < download.index("callAsync")


def test_the_launch_check_does_not_offer_a_build_that_just_failed() -> None:
    check = _body(_code(APP), "void checkForUpdates (bool userInitiated)")
    suppressed = check.index("zupdate::launchOfferSuppressed (")
    assert "! userInitiated" in check[suppressed - 40 : suppressed], "File > Check for Updates still offers it"
    assert suppressed < check.index("app->offerUpdate (")


def test_a_started_build_says_so_last_and_reads_the_last_result() -> None:
    init = _body(_code(APP), "void initialise (const juce::String& commandLine) override")
    assert "app->reportLastUpdateIfNeeded();" in init
    started = init.index("zupdate::noteStarted (")
    assert init[started - 60 : started].count("isInstalledCopy()") == 1, "a build-tree run reports nothing"
    assert started > init.index("mainWindow.reset ("), "a build that dies before its window has not started"
    assert not init[started:].count("zsplash") and init[started:].count(";") <= 2, "it is the last thing initialise does"


def test_the_self_check_saves_nothing() -> None:
    app = _code(APP)
    init = _body(app, "void initialise (const juce::String& commandLine) override")
    flag = init.index('"--self-check"')
    assert flag < init.index("crashreport::install ("), "before anything of the user's is read"
    shutdown = _body(app, "void shutdown() override")
    assert shutdown.index("if (selfCheckOnly) return;") < shutdown.index("saveUiState();")
