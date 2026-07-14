"""
Arithmetic-coding steganography codec.

Pipeline
--------
Encode (hide a message):
    1. Compress *message* to bits using the LM as a source model.
    2. Arithmetic-encode those bits into cover tokens conditioned on *context*.

Decode (recover a message):
    1. Arithmetic-decode cover tokens → bits (conditioned on same context).
    2. Decompress bits → message text using the LM as a source model.

Public API
----------
encode(tokenizer, model, message, context, *, device, ...)
    -> (cover_token_ids, cover_text)

decode(tokenizer, model, cover_token_ids, context, *, device, ...)
    -> message_text

cover_text_to_tokens(tokenizer, text)
    -> token_ids   # for cross-session decoding; BPE errors possible
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F

from .bits import bits_to_int, int_to_bits, matching_prefix_len

# Sentinel appended to the message before Stage-1 compression so the
# decompressor knows where the message ends.
_EOS = '<eos>'

# Tokens banned from every generated distribution.
# Index -1 == <|endoftext|> for GPT-2 (vocab index 50256).
# Index 628 == double-newline token.
_BANNED = (-1, 628)

# Stage-1 (message compression) parameters – high precision / wide topk
# so that any natural-language message compresses faithfully.
_S1_PRECISION = 40
_S1_TOPK = 60_000
_S1_TEMP = 1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _forward(model, tokens: torch.Tensor, past):
    """One forward pass.  Returns (next_token_logits, new_past)."""
    out = model(tokens.unsqueeze(0), past_key_values=past, use_cache=True)
    return out.logits[0, -1], out.past_key_values


def _distribution(logits: torch.Tensor, topk: int, temp: float, interval: int):
    """
    Map model logits to an integer probability distribution over [0, interval).

    Returns
    -------
    indices  : sorted vocabulary indices (descending probability)
    cum      : cumulative integer probabilities, shape (k,), in [0, interval]
    k        : effective number of tokens retained
    """
    for idx in _BANNED:
        logits[idx] = -1e20

    logits, indices = logits.sort(descending=True)
    probs = F.softmax(logits.double() / temp, dim=0)

    # Drop tokens below the rounding threshold for this interval size
    threshold = 1.0 / interval
    below = (probs < threshold).nonzero()
    k = min(max(2, below[0].item() if len(below) else topk), topk)
    probs = probs[:k]

    probs_int = (probs / probs.sum() * interval).round().long()
    cum = probs_int.cumsum(0)

    # Fix overflow from rounding
    over = (cum > interval).nonzero()
    if len(over):
        k = over[0].item()
        cum = cum[:k]

    cum = cum + (interval - cum[-1])   # absorb underflow into top bucket
    return indices, cum, k


def _bos(tokenizer) -> List[int]:
    return tokenizer.encode('<|endoftext|>', add_special_tokens=False)


def _ctx_tokens(tokenizer, context: str) -> List[int]:
    return _bos(tokenizer) + tokenizer.encode(context, add_special_tokens=False)


# ---------------------------------------------------------------------------
# BPE mismatch recovery
# ---------------------------------------------------------------------------

def _recover_rank(tokenizer, token_id: int, indices: torch.Tensor, k: int) -> int:
    """
    Find the best rank in indices[:k] when *token_id* is not present.

    Handles three common BPE round-trip mismatches:
    1. Leading/trailing space difference  (' William' vs 'William')
    2. Proposed token is a text prefix of the true token
    3. True token is a text prefix of the proposed token

    Falls back to rank 0 (most probable) if nothing matches.
    """
    true_text = tokenizer.decode([token_id])
    true_stripped = true_text.strip()

    for rank in range(k):
        prop = tokenizer.decode([indices[rank].item()])

        if true_stripped and prop.strip() == true_stripped:
            return rank                         # space-only difference

        if prop and true_text.startswith(prop):
            return rank                         # prop is a prefix

        if true_text and prop.startswith(true_text):
            return rank                         # true is a prefix

    return 0   # best-effort: most probable token


# ---------------------------------------------------------------------------
# Arithmetic coding primitives
# ---------------------------------------------------------------------------

def _interval_update(lo: int, hi: int, sel: int, cum_abs: torch.Tensor, precision: int):
    """
    Narrow [lo, hi) to the selected sub-interval and return
    (new_lo, new_hi, emitted_bits).
    """
    new_lo = cum_abs[sel - 1].item() if sel > 0 else lo
    new_hi = cum_abs[sel].item()

    lo_bits = list(reversed(int_to_bits(new_lo, precision)))
    hi_bits = list(reversed(int_to_bits(new_hi - 1, precision)))
    n = matching_prefix_len(lo_bits, hi_bits)
    emitted = hi_bits[:n]

    lo_bits = lo_bits[n:] + [0] * n
    hi_bits = hi_bits[n:] + [1] * n
    new_lo = bits_to_int(reversed(lo_bits))
    new_hi = bits_to_int(reversed(hi_bits)) + 1
    return new_lo, new_hi, emitted


def _bits_chunk(bits: List[int], offset: int, precision: int) -> int:
    """Read *precision* bits at *offset* (zero-padded) and convert to int."""
    chunk = bits[offset:offset + precision]
    chunk += [0] * max(0, precision - len(chunk))
    return bits_to_int(reversed(chunk))


# ---------------------------------------------------------------------------
# Stage 1 — compress / decompress message text
# ---------------------------------------------------------------------------

def _text_to_bits(tokenizer, model, text: str, device) -> List[int]:
    """Arithmetic-decode *text* into a bit list using the LM as source model."""
    precision, topk, temp = _S1_PRECISION, _S1_TOPK, _S1_TEMP
    lo, hi = 0, 2 ** precision

    bos_t = torch.tensor(_bos(tokenizer), device=device, dtype=torch.long)
    prev, past = bos_t, None
    bits: List[int] = []

    inp = tokenizer.encode(text + _EOS, add_special_tokens=False)

    with torch.no_grad():
        for token_id in inp:
            logits, past = _forward(model, prev, past)
            indices, cum, k = _distribution(logits, topk, temp, hi - lo)
            cum_abs = cum + lo

            t = torch.tensor(token_id, device=device)
            m = (indices[:k] == t).nonzero()
            sel = m[0].item() if len(m) else _recover_rank(tokenizer, token_id, indices, k)

            lo, hi, emitted = _interval_update(lo, hi, sel, cum_abs, precision)
            bits.extend(emitted)

            prev = torch.tensor([token_id], device=device, dtype=torch.long)

    # Flush: emit the remaining interval lower-bound so the decoder always has
    # a full precision-bit window to identify the last symbol(s).
    bits.extend(list(reversed(int_to_bits(lo, precision))))

    return bits


def _bits_to_text(tokenizer, model, bits: List[int], device) -> str:
    """Arithmetic-encode *bits* into text using the LM as source model."""
    precision, topk, temp = _S1_PRECISION, _S1_TOPK, _S1_TEMP
    lo, hi = 0, 2 ** precision

    bos_t = torch.tensor(_bos(tokenizer), device=device, dtype=torch.long)
    prev, past = bos_t, None
    out_ids: List[int] = []
    bit_pos = 0

    eos_ids = tokenizer.encode(_EOS, add_special_tokens=False)

    with torch.no_grad():
        while len(out_ids) < 512:
            logits, past = _forward(model, prev, past)
            indices, cum, k = _distribution(logits, topk, temp, hi - lo)
            cum_abs = cum + lo

            if bit_pos >= len(bits):
                sel = 0                              # bits consumed; pick top token
            else:
                msg_int = _bits_chunk(bits, bit_pos, precision)
                sel = (cum_abs > msg_int).nonzero()[0].item()
                lo, hi, emitted = _interval_update(lo, hi, sel, cum_abs, precision)
                bit_pos += len(emitted)

            token_id = indices[sel].item()
            out_ids.append(token_id)
            prev = torch.tensor([token_id], device=device, dtype=torch.long)

            if len(out_ids) >= len(eos_ids) and out_ids[-len(eos_ids):] == eos_ids:
                break

    return tokenizer.decode(out_ids, skip_special_tokens=False).replace(_EOS, '').strip()


# ---------------------------------------------------------------------------
# Stage 2 — encode bits into cover / decode cover to bits
# ---------------------------------------------------------------------------

def _encode_bits(
    tokenizer, model, bits: List[int], ctx_tokens: List[int], *,
    device, precision: int, topk: int, temp: float,
) -> List[int]:
    """Arithmetic-encode *bits* into cover token IDs conditioned on *ctx_tokens*."""
    lo, hi = 0, 2 ** precision

    ctx_t = torch.tensor(ctx_tokens[-1022:], device=device, dtype=torch.long)
    prev, past = ctx_t, None
    cover: List[int] = []
    bit_pos = 0

    with torch.no_grad():
        while bit_pos < len(bits):
            logits, past = _forward(model, prev, past)
            indices, cum, k = _distribution(logits, topk, temp, hi - lo)
            cum_abs = cum + lo

            msg_int = _bits_chunk(bits, bit_pos, precision)
            sel = (cum_abs > msg_int).nonzero()[0].item()
            lo, hi, emitted = _interval_update(lo, hi, sel, cum_abs, precision)
            bit_pos += len(emitted)

            token_id = indices[sel].item()
            cover.append(token_id)
            prev = torch.tensor([token_id], device=device, dtype=torch.long)

    return cover


def _decode_tokens(
    tokenizer, model, cover_ids: List[int], ctx_tokens: List[int], *,
    device, precision: int, topk: int, temp: float,
) -> List[int]:
    """Arithmetic-decode *cover_ids* into message bits conditioned on *ctx_tokens*."""
    lo, hi = 0, 2 ** precision

    ctx_t = torch.tensor(ctx_tokens[-1022:], device=device, dtype=torch.long)
    prev, past = ctx_t, None
    bits: List[int] = []

    with torch.no_grad():
        for token_id in cover_ids:
            logits, past = _forward(model, prev, past)
            indices, cum, k = _distribution(logits, topk, temp, hi - lo)
            cum_abs = cum + lo

            t = torch.tensor(token_id, device=device)
            m = (indices[:k] == t).nonzero()
            sel = m[0].item() if len(m) else _recover_rank(tokenizer, token_id, indices, k)

            lo, hi, emitted = _interval_update(lo, hi, sel, cum_abs, precision)
            bits.extend(emitted)

            prev = torch.tensor([token_id], device=device, dtype=torch.long)

    return bits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode(
    tokenizer,
    model,
    message: str,
    context: str,
    *,
    device,
    precision: int = 26,
    topk: int = 300,
    temp: float = 1.0,
) -> Tuple[List[int], str]:
    """
    Hide *message* in cover text conditioned on *context*.

    Returns
    -------
    cover_token_ids : list[int]
        Token IDs of the generated cover text (not including context).
        Pass these directly to :func:`decode` for lossless recovery.
    cover_text : str
        Human-readable cover text to transmit.
    """
    ctx = _ctx_tokens(tokenizer, context)
    bits = _text_to_bits(tokenizer, model, message, device)
    cover_ids = _encode_bits(tokenizer, model, bits, ctx,
                             device=device, precision=precision, topk=topk, temp=temp)
    cover_text = tokenizer.decode(cover_ids, skip_special_tokens=False)
    return cover_ids, cover_text


def decode(
    tokenizer,
    model,
    cover_token_ids: List[int],
    context: str,
    *,
    device,
    precision: int = 26,
    topk: int = 300,
    temp: float = 1.0,
) -> str:
    """
    Recover the hidden message from *cover_token_ids*.

    Parameters
    ----------
    cover_token_ids : list[int]
        Token IDs of the cover text (not including context).
        Use the value returned by :func:`encode` for lossless decoding.
        For cross-session decoding from text, convert first with
        :func:`cover_text_to_tokens`.
    context : str
        Must match the context used during encoding exactly.
    """
    ctx = _ctx_tokens(tokenizer, context)
    bits = _decode_tokens(tokenizer, model, cover_token_ids, ctx,
                          device=device, precision=precision, topk=topk, temp=temp)
    return _bits_to_text(tokenizer, model, bits, device)


def cover_text_to_tokens(tokenizer, text: str) -> List[int]:
    """
    Tokenise a cover text string into token IDs for cross-session decoding.

    BPE tokenisation may produce slightly different token boundaries than the
    originals, causing minor decoding errors.  Use the token IDs returned by
    :func:`encode` whenever possible.
    """
    return tokenizer.encode(text, add_special_tokens=False)
