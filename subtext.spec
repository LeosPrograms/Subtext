# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import collect_all

_t_d, _t_b, _t_h = collect_all('transformers')
_k_d, _k_b, _k_h = collect_all('tokenizers')
_s_d, _s_b, _s_h = collect_all('subtext')

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[*_t_b, *_k_b, *_s_b],
    datas=[
        ('model/gpt2', 'model/gpt2'),
        *_t_d, *_k_d, *_s_d,
    ],
    hiddenimports=['safetensors.torch', *_t_h, *_k_h, *_s_h],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Subtext',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=(sys.platform == 'darwin'),
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Subtext',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='Subtext.app',
        icon=None,
        bundle_identifier='com.subtext.app',
        info_plist={
            'NSHighResolutionCapable': True,
        },
    )
