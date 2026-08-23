"""
sionna_rest_server.py — Real-Sionna-backed REST twin for Scenario 6.

Run this on your OTHER workstation (the one with real Sionna installed).
It implements the exact contract scenario6_neural_rx.py's _generate_sionna_data()
already calls — so nothing on the Mac / framework side needs to change. Just:

    1. Run this script on the Sionna workstation.
    2. In config.yaml on the Mac, set:
         global.sionna_rest_url: "http://<workstation-LAN-IP>:8080"
         scenario6.use_synthetic_fallback: false
    3. Run scenario6_neural_rx.py as usual — it will fetch real data over HTTP
       and cache it to data/s6_neural_rx_dataset.npz on first run.

Contract:
    POST /api/v1/channel_samples
    body   : {"channel_model": "TDL-C", "n_samples": 2000,
              "snr_range_db": [0, 30], "modulation": "QPSK"}
    returns: {"received_symbols": [[I, Q], ...],   # shape (N, 2)
              "labels":           [int, ...]}       # shape (N,), 0-3 for QPSK

Setup on the Sionna workstation:
    pip install flask numpy
    python sionna_rest_server.py
    # Listens on 0.0.0.0:8080. Find this machine's LAN IP with:
    #   ifconfig | grep "inet " within your subnet   (macOS/Linux)

IMPORTANT — pick your backend below:
    Check what you have installed first:
        python -c "import sionna; print(sionna.__version__)"
    Sionna >= 2.0   -> PyTorch-based,   namespace is sionna.phy.channel...
    Sionna <  2.0   -> TensorFlow-based, namespace is sionna.channel... (no .phy)
    (As of Sionna 2.0, TensorFlow was dropped entirely and PHY/SYS moved to
    PyTorch; NVIDIA's release notes say the class-level API is unchanged
    across that migration, only the import path and backend tensor type.)

The two Sionna-specific functions below (_simulate_sionna_v2 /
_simulate_sionna_legacy) are written from Sionna's documented module layout,
not verified against a live install in this environment — treat them as a
strong first draft. Run BACKEND="stub" first to confirm the REST plumbing
itself works end-to-end, then switch to your real backend and paste back
the traceback if anything doesn't match your installed version; that's
faster to fix together than for me to guess every constructor argument
correctly blind.
"""

from __future__ import annotations
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

PORT = 8080

# ---------------------------------------------------------------------------
# "stub"          -> pure NumPy, no Sionna needed. Proves the contract works.
# "sionna_v2"      -> Sionna >= 2.0 (PyTorch backend)
# "sionna_legacy"  -> Sionna <  2.0 (TensorFlow backend)
# ---------------------------------------------------------------------------
BACKEND = "stub"   # <-- change this once you've confirmed the stub works

QPSK_CONSTELLATION = np.array([
    [ 1,  1], [-1,  1], [-1, -1], [ 1, -1],
], dtype=np.float32) / np.sqrt(2)


def simulate_channel(channel_model: str, n_samples: int, snr_range_db: tuple):
    """Dispatch to whichever backend is configured above."""
    if BACKEND == "sionna_v2":
        return _simulate_sionna_v2(channel_model, n_samples, snr_range_db)
    elif BACKEND == "sionna_legacy":
        return _simulate_sionna_legacy(channel_model, n_samples, snr_range_db)
    return _simulate_stub(n_samples, snr_range_db)


def _simulate_stub(n_samples, snr_range_db):
    """
    No-Sionna sanity check — same math as the framework's existing NumPy
    surrogate. This is intentionally NOT real Sionna; it only proves the
    REST wiring (request parsing, response shape, config, network path)
    works end-to-end before you touch the harder part.
    """
    rng = np.random.default_rng()
    labels = rng.integers(0, 4, size=n_samples)
    symbols = QPSK_CONSTELLATION[labels]
    snr_db = rng.uniform(snr_range_db[0], snr_range_db[1], size=n_samples).astype(np.float32)
    noise_std = 10.0 ** (-snr_db / 20.0)
    noise = rng.standard_normal((n_samples, 2)).astype(np.float32) * noise_std[:, None]
    received = (symbols + noise).astype(np.float32)
    return received, labels.astype(np.int32)


def _labels_to_bits(labels: np.ndarray) -> np.ndarray:
    """2 bits/symbol. Internally consistent — doesn't need to match any
    external Gray-code convention since we generate and score our own labels."""
    return np.stack([(labels >> 1) & 1, labels & 1], axis=-1).astype(np.float32)


def _simulate_sionna_v2(channel_model, n_samples, snr_range_db):
    """Sionna >= 2.0, PyTorch backend. UNVERIFIED against a live install — see
    module docstring. Adjust constructor args to match your version's docs."""
    import torch
    from sionna.phy.channel.tr38901 import TDL
    from sionna.phy.channel import AWGN
    from sionna.phy.mapping import Mapper

    tdl_letter = channel_model.replace("TDL-", "")
    tdl = TDL(model=tdl_letter, delay_spread=300e-9, carrier_frequency=3.5e9)
    mapper = Mapper("qam", 2)
    awgn = AWGN()

    labels = np.random.randint(0, 4, size=n_samples).astype(np.int32)
    bits = torch.tensor(_labels_to_bits(labels))
    x = mapper(bits)  # complex QPSK symbols, shape (n_samples,)

    h = tdl(batch_size=n_samples, num_time_steps=1, sampling_frequency=1e6)
    y = h[:, 0] * x

    snr_db = np.random.uniform(snr_range_db[0], snr_range_db[1], size=n_samples)
    no = 10.0 ** (-torch.tensor(snr_db, dtype=torch.float32) / 10.0)
    y = awgn(y, no)

    received = torch.stack([y.real, y.imag], dim=-1).detach().cpu().numpy().astype(np.float32)
    return received, labels


def _simulate_sionna_legacy(channel_model, n_samples, snr_range_db):
    """Pre-2.0 Sionna, TensorFlow backend. Same caveat as above."""
    import tensorflow as tf
    from sionna.channel.tr38901 import TDL
    from sionna.channel import AWGN
    from sionna.mapping import Mapper

    tdl_letter = channel_model.replace("TDL-", "")
    tdl = TDL(model=tdl_letter, delay_spread=300e-9, carrier_frequency=3.5e9)
    mapper = Mapper("qam", 2)
    awgn = AWGN()

    labels = np.random.randint(0, 4, size=n_samples).astype(np.int32)
    bits = tf.constant(_labels_to_bits(labels))
    x = mapper(bits)

    h = tdl(batch_size=n_samples, num_time_steps=1, sampling_frequency=1e6)
    y = h[:, 0] * x

    snr_db = np.random.uniform(snr_range_db[0], snr_range_db[1], size=n_samples)
    no = 10.0 ** (-tf.constant(snr_db, dtype=tf.float32) / 10.0)
    y = awgn([y, no])

    received = tf.stack([tf.math.real(y), tf.math.imag(y)], axis=-1).numpy().astype(np.float32)
    return received, labels


@app.route("/api/v1/channel_samples", methods=["POST"])
def channel_samples():
    body = request.get_json(force=True)
    channel_model = body.get("channel_model", "TDL-C")
    n_samples = int(body.get("n_samples", 2000))
    snr_range_db = tuple(body.get("snr_range_db", [0, 30]))
    modulation = body.get("modulation", "QPSK")

    if modulation != "QPSK":
        return jsonify({"error": f"only QPSK supported currently, got {modulation!r}"}), 400
    if n_samples <= 0:
        return jsonify({"error": "n_samples must be positive"}), 400

    try:
        received, labels = simulate_channel(channel_model, n_samples, snr_range_db)
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    return jsonify({
        "received_symbols": received.tolist(),
        "labels": labels.tolist(),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "backend": BACKEND})


if __name__ == "__main__":
    print(f"[sionna_rest_server] backend={BACKEND!r}  listening on 0.0.0.0:{PORT}")
    print(f"Point config.yaml's global.sionna_rest_url at this machine, e.g.:")
    print(f'  sionna_rest_url: "http://<this-machines-LAN-IP>:{PORT}"')
    app.run(host="0.0.0.0", port=PORT)
