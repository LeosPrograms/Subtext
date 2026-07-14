"""
Subtext — Neural Steganography GUI
"""

import sys
import os
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# ── Resolve model path (works both normally and inside a PyInstaller bundle) ──
_BASE = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = os.path.join(_BASE, 'model', 'gpt2')

from subtext.model import load
from subtext.codec import encode, decode, cover_text_to_tokens

# ─────────────────────────────────────────────────────────────────────────────
# Splash / model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_with_splash():
    splash = tk.Tk()
    splash.title('Subtext')
    splash.resizable(False, False)
    ttk.Label(splash, text='Loading model…', padding=(32, 16)).pack()
    bar = ttk.Progressbar(splash, mode='indeterminate', length=320)
    bar.pack(padx=32, pady=(0, 24))
    bar.start(10)
    splash.update()

    result: dict = {}

    def _do():
        try:
            result['tok'], result['mdl'], result['dev'] = load(_MODEL_PATH)
        except Exception as exc:
            result['err'] = str(exc)
        finally:
            splash.after(0, splash.destroy)

    threading.Thread(target=_do, daemon=True).start()
    splash.mainloop()

    if 'err' in result:
        tk.Tk().withdraw()
        messagebox.showerror('Subtext — load error', result['err'])
        sys.exit(1)

    return result['tok'], result['mdl'], result['dev']


tokenizer, model, device = _load_with_splash()

# ─────────────────────────────────────────────────────────────────────────────
# Session state: store the last encoded token IDs so the decode tab can
# recover the message losslessly without a BPE round-trip.
# ─────────────────────────────────────────────────────────────────────────────

_session: dict = {'cover_ids': None, 'cover_text': None, 'context': None}

DEFAULT_CONTEXT = (
    "Hi! "
    "I have been wondering whether you have made any progress on your book report from last week. "
    "If you would kindly send me what you have written so far, I would be happy to provide feedback and suggestions. "
)

# ─────────────────────────────────────────────────────────────────────────────
# Background workers
# ─────────────────────────────────────────────────────────────────────────────

def _run_encode(message, context, precision, topk, temp, on_done, on_err):
    try:
        cover_ids, cover_text = encode(
            tokenizer, model, message, context,
            device=device, precision=precision, topk=topk, temp=temp,
        )
        _session['cover_ids'] = cover_ids
        _session['cover_text'] = cover_text
        _session['context'] = context
        on_done(cover_text)
    except Exception as exc:
        on_err(str(exc))


def _run_decode(cover_ids, context, precision, topk, temp, on_done, on_err):
    try:
        message = decode(
            tokenizer, model, cover_ids, context,
            device=device, precision=precision, topk=topk, temp=temp,
        )
        on_done(message)
    except Exception as exc:
        on_err(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read(widget: scrolledtext.ScrolledText, strip_all=False) -> str:
    text = widget.get('1.0', tk.END).rstrip('\n')
    return text.strip() if strip_all else text


def _write(widget: scrolledtext.ScrolledText, text: str, readonly=False):
    """Write to a scrolledtext, respecting its disabled state."""
    if readonly:
        widget.config(state='normal')
    widget.delete('1.0', tk.END)
    widget.insert(tk.END, text)
    if readonly:
        widget.config(state='disabled')


def _set_busy(btn: ttk.Button, label: ttk.Label, msg: str):
    btn.config(state='disabled')
    label.config(text=msg)


def _set_idle(btn: ttk.Button, label: ttk.Label, msg: str):
    btn.after(0, lambda: btn.config(state='normal'))
    label.after(0, lambda: label.config(text=msg))


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

root = tk.Tk()
root.title('Subtext — Neural Steganography')
root.minsize(640, 560)

# ── Shared context ────────────────────────────────────────────────────────────
ctx_frame = ttk.LabelFrame(
    root,
    text='Shared context  (sender and receiver must use identical text)',
)
ctx_frame.pack(fill=tk.X, padx=10, pady=(10, 4))
ctx_box = scrolledtext.ScrolledText(ctx_frame, height=4, wrap=tk.WORD)
ctx_box.pack(fill=tk.X, padx=6, pady=6)
ctx_box.insert(tk.END, DEFAULT_CONTEXT)

# ── Parameters ────────────────────────────────────────────────────────────────
pf = ttk.Frame(root)
pf.pack(fill=tk.X, padx=10, pady=4)

ttk.Label(pf, text='Precision:').pack(side=tk.LEFT)
prec_var = tk.IntVar(value=26)
ttk.Spinbox(pf, from_=10, to=40, textvariable=prec_var, width=5).pack(side=tk.LEFT, padx=(2, 14))

ttk.Label(pf, text='Top-k:').pack(side=tk.LEFT)
topk_var = tk.IntVar(value=300)
ttk.Spinbox(pf, from_=50, to=60000, increment=50, textvariable=topk_var, width=7).pack(side=tk.LEFT, padx=(2, 14))

ttk.Label(pf, text='Temp:').pack(side=tk.LEFT)
temp_var = tk.DoubleVar(value=0.9)
ttk.Spinbox(pf, from_=0.1, to=2.0, increment=0.05, textvariable=temp_var,
            format='%.2f', width=6).pack(side=tk.LEFT, padx=2)

# ── Tabs ──────────────────────────────────────────────────────────────────────
nb = ttk.Notebook(root)
nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 10))

# ─── Encode tab ───────────────────────────────────────────────────────────────
enc_tab = ttk.Frame(nb)
nb.add(enc_tab, text='  Encode  ')

ttk.Label(enc_tab, text='Secret message:').pack(anchor='w', padx=6, pady=(8, 0))
msg_box = scrolledtext.ScrolledText(enc_tab, height=5, wrap=tk.WORD)
msg_box.pack(fill=tk.X, padx=6)
msg_box.insert(tk.END, 'This is a very secret message!')

ttk.Label(enc_tab, text='Cover text:').pack(anchor='w', padx=6, pady=(8, 0))
cover_out = scrolledtext.ScrolledText(enc_tab, height=7, wrap=tk.WORD, state='disabled')
cover_out.pack(fill=tk.BOTH, expand=True, padx=6)

# Buttons below the cover-text box
btn_row = ttk.Frame(enc_tab)
btn_row.pack(fill=tk.X, padx=6, pady=4)

enc_status = ttk.Label(btn_row, text='', foreground='gray')
enc_status.pack(side=tk.LEFT)

btn_to_decode = ttk.Button(btn_row, text='→ Send to Decode', state='disabled')
btn_to_decode.pack(side=tk.RIGHT, padx=(4, 0))

btn_copy = ttk.Button(btn_row, text='Copy', state='disabled')
btn_copy.pack(side=tk.RIGHT)

btn_enc = ttk.Button(enc_tab, text='Encode  ▶')
btn_enc.pack(pady=(0, 8))


def _copy_cover():
    text = _read(cover_out)
    root.clipboard_clear()
    root.clipboard_append(text)
    enc_status.config(text='Copied.')


def _send_to_decode():
    """Transfer cover text (and cached token IDs) to the Decode tab."""
    text = _read(cover_out)
    _write(cover_in, text)
    nb.select(dec_tab)


def on_encode():
    msg = _read(msg_box, strip_all=True)
    ctx = _read(ctx_box, strip_all=True)
    if not msg or not ctx:
        messagebox.showwarning('Subtext', 'Please fill in both the message and context.')
        return

    _write(cover_out, '', readonly=True)
    btn_copy.config(state='disabled')
    btn_to_decode.config(state='disabled')
    _set_busy(btn_enc, enc_status, 'Encoding…')

    def done(text):
        cover_out.after(0, lambda: _write(cover_out, text, readonly=True))
        _set_idle(btn_enc, enc_status, 'Done.')
        btn_copy.after(0, lambda: btn_copy.config(state='normal'))
        btn_to_decode.after(0, lambda: btn_to_decode.config(state='normal'))

    def err(msg):
        _set_idle(btn_enc, enc_status, 'Error.')
        btn_enc.after(0, lambda: messagebox.showerror('Encode error', msg))

    threading.Thread(
        target=_run_encode,
        args=(msg, ctx, prec_var.get(), topk_var.get(), round(temp_var.get(), 2), done, err),
        daemon=True,
    ).start()


btn_enc.config(command=on_encode)
btn_copy.config(command=_copy_cover)
btn_to_decode.config(command=_send_to_decode)

# ─── Decode tab ───────────────────────────────────────────────────────────────
dec_tab = ttk.Frame(nb)
nb.add(dec_tab, text='  Decode  ')

ttk.Label(dec_tab, text='Cover text:').pack(anchor='w', padx=6, pady=(8, 0))
cover_in = scrolledtext.ScrolledText(dec_tab, height=7, wrap=tk.WORD)
cover_in.pack(fill=tk.X, padx=6)

ttk.Label(dec_tab, text='Recovered message:').pack(anchor='w', padx=6, pady=(8, 0))
msg_out = scrolledtext.ScrolledText(dec_tab, height=5, wrap=tk.WORD, state='disabled')
msg_out.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

dec_status = ttk.Label(dec_tab, text='', foreground='gray')
dec_status.pack(anchor='w', padx=6)

btn_dec = ttk.Button(dec_tab, text='◀  Decode')
btn_dec.pack(pady=(0, 8))


def on_decode():
    # Preserve leading space — crucial for GPT-2 BPE token boundaries
    cover_text = _read(cover_in)
    ctx = _read(ctx_box, strip_all=True)
    if not cover_text.strip() or not ctx:
        messagebox.showwarning('Subtext', 'Please fill in both the cover text and context.')
        return

    _write(msg_out, '', readonly=True)
    _set_busy(btn_dec, dec_status, 'Decoding…')

    # Prefer cached token IDs from the session (no BPE round-trip errors).
    # Fall back to re-tokenising the text for cross-session decoding.
    if (
        _session['cover_ids'] is not None
        and _session['cover_text'] is not None
        and _session['cover_text'].strip() == cover_text.strip()
        and _session['context'] == ctx
    ):
        cover_ids = _session['cover_ids']
        dec_status.config(text='Decoding (session cache)…')
    else:
        cover_ids = cover_text_to_tokens(tokenizer, cover_text)
        dec_status.config(text='Decoding (re-tokenised)…')

    def done(msg):
        msg_out.after(0, lambda: _write(msg_out, msg, readonly=True))
        _set_idle(btn_dec, dec_status, 'Done.')

    def err(msg):
        _set_idle(btn_dec, dec_status, 'Error.')
        btn_dec.after(0, lambda: messagebox.showerror('Decode error', msg))

    threading.Thread(
        target=_run_decode,
        args=(cover_ids, ctx, prec_var.get(), topk_var.get(), round(temp_var.get(), 2), done, err),
        daemon=True,
    ).start()


btn_dec.config(command=on_decode)

root.mainloop()
