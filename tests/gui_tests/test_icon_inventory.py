# -*- coding: utf-8 -*-
"""
Tests for Third-Party Notices and SVG Icon Inventory.

Task: T-018.7 (ADR-018 §T-018.7, §NF-18.6)
Validates that:
1. THIRD_PARTY_NOTICES.md exists in repo root with complete Font Awesome Free 7.1.0
   licensing notices (CC BY 4.0 for icons, SIL OFL 1.1 for fonts, MIT for code),
   attributions, copyright notices, license links, and the inventory table.
2. All SVG icon filenames referenced across the codebase in src/ exist in
   src/ui/rendering/icons/ (with documented legacy fallback probe handling).
3. All shipped SVGs are non-empty, parse as valid XML with <svg> root element,
   have normalized viewBox="0 0 640 640", and contain Font Awesome license headers.
4. The contract for triangle-exclamation-solid-full.svg is asserted (non-empty, valid
   XML with <svg> root element; xfails cleanly in Wave 0 until delivered by Wave 1 T-018.8).
5. All shipped icons load cleanly into QIcon and render non-null pixmaps.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ICONS_DIR = REPO_ROOT / "src" / "ui" / "rendering" / "icons"
SRC_DIR = REPO_ROOT / "src"
NOTICES_FILE = REPO_ROOT / "THIRD_PARTY_NOTICES.md"

# Canonical inventory of shipped SVG icon files in Wave 0
SHIPPED_ICONS = frozenset({
    "bullseye-solid-full.svg",
    "check-solid-full.svg",
    "circle-chevron-left-solid-full.svg",
    "circle-chevron-right-solid-full.svg",
    "floppy-disk-solid-full.svg",
    "folder-open-solid-full.svg",
    "gear-solid-full.svg",
    "image-solid-full.svg",
    "repeat-solid-full.svg",
})


# ==============================================================================
# 1. Shipped Icons Presence and Non-Empty Checks
# ==============================================================================

def test_icons_directory_exists():
    """Verify that the icons directory exists and is a directory."""
    assert ICONS_DIR.exists(), f"Icons directory not found at {ICONS_DIR}"
    assert ICONS_DIR.is_dir(), f"{ICONS_DIR} must be a directory"


@pytest.mark.parametrize("icon_filename", sorted(SHIPPED_ICONS))
def test_shipped_icon_files_exist_and_non_empty(icon_filename: str):
    """Verify that each shipped SVG icon exists, is a regular file, and is non-empty."""
    icon_path = ICONS_DIR / icon_filename
    assert icon_path.exists(), f"Shipped icon missing: {icon_filename} at {icon_path}"
    assert icon_path.is_file(), f"{icon_filename} must be a regular file"
    file_size = icon_path.stat().st_size
    assert file_size > 0, f"{icon_filename} is empty (0 bytes)"
    assert file_size >= 100, f"{icon_filename} suspiciously small ({file_size} bytes)"


# ==============================================================================
# 2. XML / SVG Validity and Header Checks
# ==============================================================================

@pytest.mark.parametrize("icon_filename", sorted(SHIPPED_ICONS))
def test_shipped_icons_are_valid_svg_xml(icon_filename: str):
    """Verify that each shipped icon parses as valid XML with an <svg> root element."""
    icon_path = ICONS_DIR / icon_filename
    try:
        tree = ET.parse(icon_path)
    except ET.ParseError as err:
        pytest.fail(f"Failed to parse XML for {icon_filename}: {err}")

    root = tree.getroot()
    tag_local = root.tag.split("}")[-1] if "}" in root.tag else root.tag
    assert tag_local == "svg", f"{icon_filename} root element must be <svg>, got <{root.tag}>"

    # All shipped Font Awesome Free 7.1.0 icons in this project are normalized to 640x640 viewBox
    viewbox = root.attrib.get("viewBox")
    assert viewbox == "0 0 640 640", (
        f"{icon_filename} viewBox must be '0 0 640 640', got '{viewbox}'"
    )

    # Must contain at least one visual child element (e.g. <path> or <g>)
    children = list(root)
    assert len(children) > 0, f"{icon_filename} has no visual SVG child elements"


@pytest.mark.parametrize("icon_filename", sorted(SHIPPED_ICONS))
def test_shipped_icons_license_header(icon_filename: str):
    """Verify that each shipped icon contains the Font Awesome Free 7.1.0 license header."""
    icon_path = ICONS_DIR / icon_filename
    raw_content = icon_path.read_text(encoding="utf-8")

    assert "Font Awesome Free 7.1.0" in raw_content, (
        f"{icon_filename} missing Font Awesome Free 7.1.0 header comment"
    )
    assert "https://fontawesome.com/license/free" in raw_content, (
        f"{icon_filename} missing license link comment"
    )
    assert "Fonticons, Inc." in raw_content, (
        f"{icon_filename} missing Fonticons, Inc. copyright attribution"
    )


def test_no_untracked_stray_files_in_icons_dir():
    """Verify that all files in src/ui/rendering/icons/ are valid SVG files."""
    for entry in ICONS_DIR.iterdir():
        if entry.is_file():
            assert entry.suffix.lower() == ".svg", (
                f"Unexpected non-SVG file in icons dir: {entry.name}"
            )


# ==============================================================================
# 3. Codebase Reference Verification
# ==============================================================================

def _scan_codebase_svg_references() -> dict[str, list[tuple[str, int]]]:
    """Scan all Python files in src/ for string literals ending in .svg."""
    svg_pattern = re.compile(r"['\"]([a-zA-Z0-9_\-\.]+\.svg)['\"]")
    references: dict[str, list[tuple[str, int]]] = {}

    for root, _, files in os.walk(SRC_DIR):
        for f in sorted(files):
            if f.endswith(".py"):
                file_path = Path(root) / f
                rel_path = file_path.relative_to(REPO_ROOT).as_posix()
                try:
                    content = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue

                for line_idx, line in enumerate(content.splitlines(), start=1):
                    for match in svg_pattern.findall(line):
                        references.setdefault(match, []).append((rel_path, line_idx))

    return references


def test_codebase_referenced_svg_icons_exist():
    """
    Verify that all SVG icon filenames referenced across the codebase exist
    in src/ui/rendering/icons/.
    """
    references = _scan_codebase_svg_references()
    assert references, "Expected to find SVG icon references in src/"

    # Legacy absent icons documented in ADR-014/ADR-018:
    # warning-solid.svg is probed in file_list_controller.py with a safe fallback to image-solid-full.svg.
    # ADR-018 acceptance specifies replacing it with triangle-exclamation-solid-full.svg in Wave 1.
    known_legacy_fallback_probes = {"warning-solid.svg"}

    missing_icons: dict[str, list[tuple[str, int]]] = {}

    for icon_name, locations in references.items():
        if icon_name in known_legacy_fallback_probes:
            continue
        icon_path = ICONS_DIR / icon_name
        if not icon_path.exists():
            missing_icons[icon_name] = locations

    assert not missing_icons, (
        f"The following SVG icons are referenced in src/ but missing from {ICONS_DIR}:\n"
        + "\n".join(f"  - {icon}: {locs}" for icon, locs in missing_icons.items())
    )


def test_all_shipped_icons_are_referenced_in_codebase():
    """Verify that all 9 shipped icons in the inventory are actually used in the codebase."""
    references = _scan_codebase_svg_references()
    referenced_names = set(references.keys())

    for shipped in SHIPPED_ICONS:
        assert shipped in referenced_names, (
            f"Shipped icon {shipped} is in src/ui/rendering/icons/ but never referenced in src/"
        )


# ==============================================================================
# 4. Specific Assertion on triangle-exclamation-solid-full.svg (ADR-018 / T-018.7)
# ==============================================================================

def test_triangle_exclamation_icon_spec():
    """
    Specifically assert triangle-exclamation-solid-full.svg and all shipped icon files
    exist, are valid XML/SVG with <svg> root element, and non-empty.

    In Wave 0, triangle-exclamation-solid-full.svg is scheduled for addition in Wave 1
    (T-018.8 per ADR-018 §T-018.8). If the file is not yet present on disk, this test
    cleanly marks xfail so that Wave 0 test suites pass while preserving the strict
    contract verification for when Wave 1 lands.
    """
    # 1. Assert all shipped icon files exist, are valid XML/SVG, and non-empty
    for shipped in sorted(SHIPPED_ICONS):
        p = ICONS_DIR / shipped
        assert p.exists(), f"Shipped icon {shipped} must exist"
        assert p.stat().st_size > 0, f"Shipped icon {shipped} must be non-empty"
        tree = ET.parse(p)
        tag_local = tree.getroot().tag.split("}")[-1]
        assert tag_local == "svg", f"{shipped} root element must be <svg>"

    # 2. Specifically assert triangle-exclamation-solid-full.svg
    target_icon = ICONS_DIR / "triangle-exclamation-solid-full.svg"
    if not target_icon.exists():
        pytest.xfail(
            "triangle-exclamation-solid-full.svg is scheduled for delivery in Wave 1 "
            "(T-018.8 per ADR-018 §T-018.8). Unblocked by T-018.7."
        )

    # When present (in Wave 1+), verify full contract
    assert target_icon.is_file(), f"{target_icon.name} must be a regular file"
    assert target_icon.stat().st_size > 0, f"{target_icon.name} must be non-empty"
    tree = ET.parse(target_icon)
    root = tree.getroot()
    tag_local = root.tag.split("}")[-1] if "}" in root.tag else root.tag
    assert tag_local == "svg", f"{target_icon.name} root element must be <svg>, got <{root.tag}>"
    assert root.attrib.get("viewBox") == "0 0 640 640", (
        f"{target_icon.name} must have viewBox='0 0 640 640'"
    )


# ==============================================================================
# 5. THIRD_PARTY_NOTICES.md Documentation Compliance
# ==============================================================================

def test_third_party_notices_file_compliance():
    """Verify that THIRD_PARTY_NOTICES.md exists, is non-empty, and contains all required notices."""
    assert NOTICES_FILE.exists(), f"THIRD_PARTY_NOTICES.md missing from {REPO_ROOT}"
    assert NOTICES_FILE.is_file(), "THIRD_PARTY_NOTICES.md must be a regular file"
    assert NOTICES_FILE.stat().st_size > 500, "THIRD_PARTY_NOTICES.md suspiciously short"

    content = NOTICES_FILE.read_text(encoding="utf-8")

    # Font Awesome Free 7.1.0 and creator attribution
    assert "Font Awesome Free 7.1.0" in content
    assert "Fonticons, Inc." in content
    assert "https://fontawesome.com" in content
    assert "https://fontawesome.com/license/free" in content

    # Triple licensing terms
    assert "CC BY 4.0" in content or "Creative Commons Attribution 4.0" in content
    assert "SIL OFL 1.1" in content or "SIL Open Font License" in content
    assert "MIT License" in content

    # Shipped inventory entries
    for icon_filename in sorted(SHIPPED_ICONS):
        assert icon_filename in content, (
            f"Shipped icon {icon_filename} not listed in THIRD_PARTY_NOTICES.md inventory"
        )


# ==============================================================================
# 6. Qt QIcon Rendering Integration
# ==============================================================================

def test_qicon_loads_shipped_icons(qapp):
    """Verify that all shipped SVGs can be loaded by PyQt6.QtGui.QIcon and render non-null pixmaps."""
    from PyQt6.QtCore import QSize
    from PyQt6.QtGui import QIcon

    for icon_filename in sorted(SHIPPED_ICONS):
        icon_path = ICONS_DIR / icon_filename
        icon = QIcon(str(icon_path))
        assert not icon.isNull(), f"QIcon failed to load {icon_filename}"

        pixmap = icon.pixmap(QSize(24, 24))
        assert not pixmap.isNull(), f"QIcon rendered null pixmap for {icon_filename}"
        assert pixmap.width() > 0 and pixmap.height() > 0
