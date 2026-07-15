#!/usr/bin/env bash
# build-appimage.sh — Build a Linux AppImage for Subtext
set -euo pipefail

APPDIR="AppDir"
APPIMAGETOOL_URL="https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
APPIMAGETOOL="./appimagetool-x86_64.AppImage"

# ── 1. Build the PyInstaller one-file binary ──────────────────────────────────
echo ">>> Building with PyInstaller..."
pyinstaller subtext.spec

# ── 2. Populate AppDir ────────────────────────────────────────────────────────
echo ">>> Populating AppDir..."
cp dist/Subtext "$APPDIR/Subtext"
chmod +x "$APPDIR/Subtext"
chmod +x "$APPDIR/AppRun"

# ── 3. Icon (required by appimagetool) ────────────────────────────────────────
# If a real icon exists, copy it; otherwise generate a minimal placeholder.
if [ -f "subtext.png" ]; then
    cp subtext.png "$APPDIR/subtext.png"
elif ! [ -f "$APPDIR/subtext.png" ]; then
    echo ">>> No icon found — generating a placeholder (requires Python + Pillow)..."
    python3 - <<'PYEOF'
try:
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (256, 256), (30, 30, 46, 255))
    d = ImageDraw.Draw(img)
    d.ellipse([32, 32, 224, 224], fill=(137, 180, 250, 255))
    img.save("AppDir/subtext.png")
    print("Placeholder icon written to AppDir/subtext.png")
except ImportError:
    # Absolute fallback: a 1×1 transparent PNG
    import base64, pathlib
    tiny = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
        "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    pathlib.Path("AppDir/subtext.png").write_bytes(tiny)
    print("Minimal placeholder icon written to AppDir/subtext.png")
PYEOF
fi

# ── 4. Download appimagetool if needed ────────────────────────────────────────
if [ ! -f "$APPIMAGETOOL" ]; then
    echo ">>> Downloading appimagetool..."
    curl -fsSL -o "$APPIMAGETOOL" "$APPIMAGETOOL_URL"
    chmod +x "$APPIMAGETOOL"
fi

# ── 5. Package as AppImage ────────────────────────────────────────────────────
echo ">>> Creating AppImage..."
ARCH=x86_64 "$APPIMAGETOOL" "$APPDIR" Subtext-x86_64.AppImage

echo ""
echo "Done! Output: Subtext-x86_64.AppImage"
