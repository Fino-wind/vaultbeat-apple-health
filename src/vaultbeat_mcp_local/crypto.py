from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


ENVELOPE_INFO = b"tether-e2ee-envelope"


class VaultbeatCryptoError(ValueError):
    pass


@dataclass(frozen=True)
class RecipientKey:
    """A recipient an envelope is sealed for, with its raw Curve25519 public key (base64)."""

    recipient_kind: str
    recipient_id: str
    public_key_base64: str


@dataclass(frozen=True)
class SealedEnvelope:
    """One sealed envelope per recipient."""

    recipient_kind: str
    recipient_id: str
    encrypted_data_key_base64: str


def _b64decode(value: str, label: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except Exception as error:
        raise VaultbeatCryptoError(f"invalid base64 field: {label}") from error


def _split_apple_aes_gcm_combined(combined: bytes) -> tuple[bytes, bytes]:
    if len(combined) < 12 + 16:
        raise VaultbeatCryptoError("invalid AES-GCM combined payload")
    nonce = combined[:12]
    ciphertext_and_tag = combined[12:]
    return nonce, ciphertext_and_tag


def _seal_apple_aes_gcm_combined(key: bytes, plaintext: bytes) -> bytes:
    """AES-GCM seal in Apple's "combined" layout: nonce[12] || ciphertext || tag[16].

    Mirrors Swift CryptoKit `AES.GCM.seal(...).combined`, which is what
    `_split_apple_aes_gcm_combined` (and iOS) expect on the decrypt side. A fresh random
    96-bit nonce per call, exactly as CryptoKit generates internally.
    """

    nonce = os.urandom(12)
    return nonce + AESGCM(key).encrypt(nonce, plaintext, None)


def generate_x25519_keypair() -> tuple[str, str]:
    private_key = x25519.X25519PrivateKey.generate()
    private_raw = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(private_raw).decode(), base64.b64encode(public_raw).decode()


def public_key_from_private(private_key_base64: str) -> str:
    private_raw = _b64decode(private_key_base64, "private_key_base64")
    if len(private_raw) != 32:
        raise VaultbeatCryptoError("X25519 private key must be 32 bytes")
    private_key = x25519.X25519PrivateKey.from_private_bytes(private_raw)
    public_raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(public_raw).decode()


def decrypt_blob_payload(
    *,
    ciphertext_base64: str,
    encrypted_data_key_base64: str,
    private_key_base64: str,
) -> bytes:
    """Decrypt one E2EE blob to plaintext bytes.

    The transport (Curve25519 ECDH + HKDF-SHA256 salt="" info="tether-e2ee-envelope"
    + AES-GCM) is identical for every health kind (sleep / menstrual / water); only the
    post-decrypt JSON decode differs per metric_type, which the service layer routes.
    """

    private_raw = _b64decode(private_key_base64, "private_key_base64")
    if len(private_raw) != 32:
        raise VaultbeatCryptoError("X25519 private key must be 32 bytes")

    envelope_data = _b64decode(encrypted_data_key_base64, "encrypted_data_key")
    try:
        envelope = json.loads(envelope_data)
    except json.JSONDecodeError as error:
        raise VaultbeatCryptoError("encrypted_data_key is not a JSON envelope") from error

    if not isinstance(envelope, dict):
        raise VaultbeatCryptoError("encrypted_data_key envelope must be an object")

    sender_public_key_base64 = envelope.get("senderPublicKeyBase64")
    wrapped_symmetric_key_base64 = envelope.get("wrappedSymmetricKeyBase64")
    if not isinstance(sender_public_key_base64, str) or not isinstance(wrapped_symmetric_key_base64, str):
        raise VaultbeatCryptoError("encrypted_data_key envelope is missing key material")

    sender_public_raw = _b64decode(sender_public_key_base64, "senderPublicKeyBase64")
    if len(sender_public_raw) != 32:
        raise VaultbeatCryptoError("sender public key must be 32 bytes")

    private_key = x25519.X25519PrivateKey.from_private_bytes(private_raw)
    sender_public_key = x25519.X25519PublicKey.from_public_bytes(sender_public_raw)
    shared_secret = private_key.exchange(sender_public_key)
    wrapping_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"",
        info=ENVELOPE_INFO,
    ).derive(shared_secret)

    # AESGCM.decrypt raises cryptography.exceptions.InvalidTag on any auth
    # failure (tampered ciphertext, rotated key, envelope/recipient mismatch).
    # InvalidTag subclasses neither ValueError nor VaultbeatCryptoError, so if it
    # escaped here it would skip every per-record `except (…, VaultbeatCryptoError,
    # …)` in service.py and one bad envelope would abort the ENTIRE sync across
    # all metric types instead of landing in that record's errors[] entry.
    # Normalize it at this boundary like every other crypto-layer failure.
    try:
        wrapped_dek_combined = _b64decode(wrapped_symmetric_key_base64, "wrappedSymmetricKeyBase64")
        wrapped_nonce, wrapped_ciphertext = _split_apple_aes_gcm_combined(wrapped_dek_combined)
        dek = AESGCM(wrapping_key).decrypt(wrapped_nonce, wrapped_ciphertext, None)

        ciphertext_combined = _b64decode(ciphertext_base64, "ciphertext")
        ciphertext_nonce, ciphertext = _split_apple_aes_gcm_combined(ciphertext_combined)
        return AESGCM(dek).decrypt(ciphertext_nonce, ciphertext, None)
    except InvalidTag as error:
        raise VaultbeatCryptoError("AES-GCM authentication failed (tampered or mis-keyed envelope)") from error


def encrypt_blob_payload(
    *,
    plaintext: bytes,
    recipients: list[RecipientKey],
) -> tuple[str, list[SealedEnvelope]]:
    """Seal `plaintext` into (ciphertext_base64, per-recipient envelopes).

    The inverse of ``decrypt_blob_payload`` and byte-compatible with iOS
    ``EnvelopeBuilder`` (verified by the round-trip tests):

    - One fresh random AES-256 DEK encrypts the payload once (Apple combined layout),
      shared across all recipients.
    - Each recipient gets the DEK wrapped under an ECDH+HKDF key derived from a FRESH
      per-envelope EPHEMERAL Curve25519 sender keypair — NOT a static server key. This
      matches iOS (`Curve25519.KeyAgreement.PrivateKey()` per envelope) so an
      iOS-side or Python-side decrypt resolves either direction.
    - The envelope wire shape is base64(JSON{senderPublicKeyBase64, wrappedSymmetricKeyBase64}),
      identical to what ``decrypt_blob_payload`` parses.

    The envelope id is intentionally NOT computed here: the server
    derives it server-side from (blob_id, recipient_kind, recipient_id) so it can never
    be a client-chosen random value (AGENTS.md anti-pattern 18).
    """

    if not recipients:
        raise VaultbeatCryptoError("encrypt_blob_payload requires at least one recipient")

    dek = os.urandom(32)
    ciphertext_base64 = base64.b64encode(_seal_apple_aes_gcm_combined(dek, plaintext)).decode()

    envelopes: list[SealedEnvelope] = []
    for recipient in recipients:
        recipient_public_raw = _b64decode(recipient.public_key_base64, "recipient public_key")
        if len(recipient_public_raw) != 32:
            raise VaultbeatCryptoError("recipient public key must be 32 bytes")

        ephemeral_private = x25519.X25519PrivateKey.generate()
        recipient_public = x25519.X25519PublicKey.from_public_bytes(recipient_public_raw)
        shared_secret = ephemeral_private.exchange(recipient_public)
        wrapping_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"",
            info=ENVELOPE_INFO,
        ).derive(shared_secret)

        wrapped_dek = _seal_apple_aes_gcm_combined(wrapping_key, dek)
        ephemeral_public_raw = ephemeral_private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        envelope_json = json.dumps(
            {
                "senderPublicKeyBase64": base64.b64encode(ephemeral_public_raw).decode(),
                "wrappedSymmetricKeyBase64": base64.b64encode(wrapped_dek).decode(),
            },
            separators=(",", ":"),
        )
        envelopes.append(
            SealedEnvelope(
                recipient_kind=recipient.recipient_kind,
                recipient_id=recipient.recipient_id,
                encrypted_data_key_base64=base64.b64encode(envelope_json.encode()).decode(),
            )
        )

    return ciphertext_base64, envelopes


def decode_json_payload(payload: bytes) -> Any:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return payload.decode("utf-8", errors="replace")


# Backwards-compatible alias: the decryption is metric-agnostic, but the original name
# predates the menstrual/water kinds. Keep it so callers importing the old name still work.
decrypt_sleep_payload = decrypt_blob_payload
