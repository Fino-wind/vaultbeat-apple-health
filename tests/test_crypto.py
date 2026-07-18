from __future__ import annotations

import base64
import json

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from vaultbeat_mcp_local.crypto import (
    ENVELOPE_INFO,
    RecipientKey,
    VaultbeatCryptoError,
    decrypt_blob_payload,
    decrypt_sleep_payload,
    encrypt_blob_payload,
    generate_x25519_keypair,
    public_key_from_private,
)


def _apple_combined(nonce: bytes, ciphertext_and_tag: bytes) -> bytes:
    return nonce + ciphertext_and_tag


def test_decrypt_sleep_payload_matches_ios_envelope_format() -> None:
    recipient_private_base64, recipient_public_base64 = generate_x25519_keypair()
    recipient_public = x25519.X25519PublicKey.from_public_bytes(base64.b64decode(recipient_public_base64))
    sender_private = x25519.X25519PrivateKey.generate()

    shared_secret = sender_private.exchange(recipient_public)
    wrapping_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"",
        info=ENVELOPE_INFO,
    ).derive(shared_secret)

    dek = b"\x04" * 32
    wrapped_nonce = b"\x01" * 12
    wrapped_dek = _apple_combined(wrapped_nonce, AESGCM(wrapping_key).encrypt(wrapped_nonce, dek, None))
    sender_public_raw = sender_private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    envelope = {
        "senderPublicKeyBase64": base64.b64encode(sender_public_raw).decode(),
        "wrappedSymmetricKeyBase64": base64.b64encode(wrapped_dek).decode(),
    }
    encrypted_data_key_base64 = base64.b64encode(json.dumps(envelope).encode()).decode()

    plaintext = b'{"sleep":"ok"}'
    ciphertext_nonce = b"\x02" * 12
    ciphertext = _apple_combined(ciphertext_nonce, AESGCM(dek).encrypt(ciphertext_nonce, plaintext, None))

    decrypted = decrypt_sleep_payload(
        ciphertext_base64=base64.b64encode(ciphertext).decode(),
        encrypted_data_key_base64=encrypted_data_key_base64,
        private_key_base64=recipient_private_base64,
    )

    assert decrypted == plaintext
    assert len(base64.b64decode(recipient_private_base64)) == 32


def test_encrypt_blob_payload_round_trips_through_decrypt() -> None:
    recipient_private_base64, recipient_public_base64 = generate_x25519_keypair()
    plaintext = b'{"description":"dark circles","content":"worse with late nights"}'

    ciphertext_base64, envelopes = encrypt_blob_payload(
        plaintext=plaintext,
        recipients=[RecipientKey("mcp_server", "srv-1", recipient_public_base64)],
    )

    assert len(envelopes) == 1
    assert envelopes[0].recipient_kind == "mcp_server"
    assert envelopes[0].recipient_id == "srv-1"
    decrypted = decrypt_blob_payload(
        ciphertext_base64=ciphertext_base64,
        encrypted_data_key_base64=envelopes[0].encrypted_data_key_base64,
        private_key_base64=recipient_private_base64,
    )
    assert decrypted == plaintext


def test_encrypt_blob_payload_multi_recipient_shares_one_dek_with_distinct_ephemerals() -> None:
    a_private, a_public = generate_x25519_keypair()
    b_private, b_public = generate_x25519_keypair()
    plaintext = b"shared fact body"

    ciphertext_base64, envelopes = encrypt_blob_payload(
        plaintext=plaintext,
        recipients=[
            RecipientKey("owner_user", "u-a", a_public),
            RecipientKey("mcp_server", "srv-b", b_public),
        ],
    )

    by_kind = {envelope.recipient_kind: envelope for envelope in envelopes}
    assert set(by_kind) == {"owner_user", "mcp_server"}

    # Each recipient unwraps the SAME dek from the SAME single ciphertext.
    for private_key, kind in ((a_private, "owner_user"), (b_private, "mcp_server")):
        assert (
            decrypt_blob_payload(
                ciphertext_base64=ciphertext_base64,
                encrypted_data_key_base64=by_kind[kind].encrypted_data_key_base64,
                private_key_base64=private_key,
            )
            == plaintext
        )

    # Sender key is EPHEMERAL per envelope (matches iOS) — never reused, never static.
    sender_a = json.loads(base64.b64decode(by_kind["owner_user"].encrypted_data_key_base64))["senderPublicKeyBase64"]
    sender_b = json.loads(base64.b64decode(by_kind["mcp_server"].encrypted_data_key_base64))["senderPublicKeyBase64"]
    assert sender_a != sender_b


def test_encrypt_blob_payload_rejects_no_recipients_and_bad_pubkey() -> None:
    try:
        encrypt_blob_payload(plaintext=b"x", recipients=[])
    except VaultbeatCryptoError:
        pass
    else:
        raise AssertionError("expected VaultbeatCryptoError for empty recipients")

    bad = RecipientKey("mcp_server", "s", base64.b64encode(b"too-short").decode())
    try:
        encrypt_blob_payload(plaintext=b"x", recipients=[bad])
    except VaultbeatCryptoError:
        pass
    else:
        raise AssertionError("expected VaultbeatCryptoError for bad recipient public key")


# ---------------------------------------------------------------------------
# Negative tests: truncated / too-short combined ciphertext
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "short_payload",
    [
        b"",          # 0 bytes — way below nonce(12)+tag(16)=28
        b"\x00" * 5,  # 5 bytes — below threshold
        b"\x00" * 27, # 27 bytes — one byte below threshold
    ],
    ids=["empty", "5-bytes", "27-bytes"],
)
def test_decrypt_blob_payload_rejects_truncated_ciphertext(short_payload: bytes) -> None:
    """_split_apple_aes_gcm_combined raises VaultbeatCryptoError when len < 28."""
    priv, pub = generate_x25519_keypair()
    _, envelopes = encrypt_blob_payload(
        plaintext=b"legit payload",
        recipients=[RecipientKey("owner_user", "u-1", pub)],
    )
    truncated_b64 = base64.b64encode(short_payload).decode()
    with pytest.raises(VaultbeatCryptoError, match="invalid AES-GCM combined payload"):
        decrypt_blob_payload(
            ciphertext_base64=truncated_b64,
            encrypted_data_key_base64=envelopes[0].encrypted_data_key_base64,
            private_key_base64=priv,
        )


def test_decrypt_blob_payload_rejects_truncated_wrapped_dek() -> None:
    """_split_apple_aes_gcm_combined raises VaultbeatCryptoError on short wrappedSymmetricKey."""
    priv, pub = generate_x25519_keypair()
    ct_b64, envelopes = encrypt_blob_payload(
        plaintext=b"legit payload",
        recipients=[RecipientKey("owner_user", "u-1", pub)],
    )
    env_json = json.loads(base64.b64decode(envelopes[0].encrypted_data_key_base64))
    env_json["wrappedSymmetricKeyBase64"] = base64.b64encode(b"tooshort").decode()
    tampered_env_b64 = base64.b64encode(json.dumps(env_json).encode()).decode()

    with pytest.raises(VaultbeatCryptoError, match="invalid AES-GCM combined payload"):
        decrypt_blob_payload(
            ciphertext_base64=ct_b64,
            encrypted_data_key_base64=tampered_env_b64,
            private_key_base64=priv,
        )


# ---------------------------------------------------------------------------
# Negative tests: wrong-length keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_private_b64",
    [
        base64.b64encode(b"").decode(),          # 0 bytes
        base64.b64encode(b"short").decode(),      # 5 bytes
        base64.b64encode(b"\x00" * 16).decode(),  # 16 bytes
        base64.b64encode(b"\x00" * 33).decode(),  # 33 bytes — one over
    ],
    ids=["0-bytes", "5-bytes", "16-bytes", "33-bytes"],
)
def test_decrypt_blob_payload_rejects_wrong_length_private_key(bad_private_b64: str) -> None:
    """crypto.py line 108: raises VaultbeatCryptoError when private key != 32 bytes."""
    priv, pub = generate_x25519_keypair()
    ct_b64, envelopes = encrypt_blob_payload(
        plaintext=b"x",
        recipients=[RecipientKey("owner_user", "u-1", pub)],
    )
    with pytest.raises(VaultbeatCryptoError, match="X25519 private key must be 32 bytes"):
        decrypt_blob_payload(
            ciphertext_base64=ct_b64,
            encrypted_data_key_base64=envelopes[0].encrypted_data_key_base64,
            private_key_base64=bad_private_b64,
        )


@pytest.mark.parametrize(
    "bad_private_b64",
    [
        base64.b64encode(b"").decode(),
        base64.b64encode(b"\x00" * 16).decode(),
        base64.b64encode(b"\x00" * 31).decode(),
    ],
    ids=["0-bytes", "16-bytes", "31-bytes"],
)
def test_public_key_from_private_rejects_wrong_length(bad_private_b64: str) -> None:
    """public_key_from_private line 84: raises VaultbeatCryptoError when private key != 32 bytes."""
    with pytest.raises(VaultbeatCryptoError, match="X25519 private key must be 32 bytes"):
        public_key_from_private(bad_private_b64)


@pytest.mark.parametrize(
    "bad_recipient_pub_b64",
    [
        base64.b64encode(b"").decode(),
        base64.b64encode(b"not32bytes").decode(),
        base64.b64encode(b"\x00" * 31).decode(),
    ],
    ids=["0-bytes", "10-bytes", "31-bytes"],
)
def test_encrypt_blob_payload_rejects_wrong_length_recipient_pubkey(bad_recipient_pub_b64: str) -> None:
    """encrypt_blob_payload line 181: raises VaultbeatCryptoError when recipient public key != 32 bytes."""
    with pytest.raises(VaultbeatCryptoError, match="recipient public key must be 32 bytes"):
        encrypt_blob_payload(
            plaintext=b"x",
            recipients=[RecipientKey("owner_user", "u-1", bad_recipient_pub_b64)],
        )


def test_decrypt_blob_payload_rejects_wrong_length_sender_pubkey_in_envelope() -> None:
    """crypto.py line 126: raises VaultbeatCryptoError when senderPublicKeyBase64 != 32 bytes."""
    priv, pub = generate_x25519_keypair()
    ct_b64, envelopes = encrypt_blob_payload(
        plaintext=b"x",
        recipients=[RecipientKey("owner_user", "u-1", pub)],
    )
    env_json = json.loads(base64.b64decode(envelopes[0].encrypted_data_key_base64))
    env_json["senderPublicKeyBase64"] = base64.b64encode(b"not32bytes").decode()
    tampered_env_b64 = base64.b64encode(json.dumps(env_json).encode()).decode()

    with pytest.raises(VaultbeatCryptoError, match="sender public key must be 32 bytes"):
        decrypt_blob_payload(
            ciphertext_base64=ct_b64,
            encrypted_data_key_base64=tampered_env_b64,
            private_key_base64=priv,
        )


# ---------------------------------------------------------------------------
# Negative tests: tampered AES-GCM tag
# ---------------------------------------------------------------------------


def test_decrypt_blob_payload_raises_on_tampered_ciphertext_tag() -> None:
    """A flipped ciphertext tag surfaces as VaultbeatCryptoError (InvalidTag is normalized at the crypto boundary so one bad envelope cannot abort a whole sync)."""
    priv, pub = generate_x25519_keypair()
    ct_b64, envelopes = encrypt_blob_payload(
        plaintext=b"secret data",
        recipients=[RecipientKey("owner_user", "u-1", pub)],
    )
    ct = base64.b64decode(ct_b64)
    tampered_ct = ct[:-1] + bytes([ct[-1] ^ 0xFF])
    tampered_ct_b64 = base64.b64encode(tampered_ct).decode()

    with pytest.raises(VaultbeatCryptoError):
        decrypt_blob_payload(
            ciphertext_base64=tampered_ct_b64,
            encrypted_data_key_base64=envelopes[0].encrypted_data_key_base64,
            private_key_base64=priv,
        )


def test_decrypt_blob_payload_raises_on_tampered_wrapped_dek_tag() -> None:
    """A flipped wrapped-DEK tag surfaces as VaultbeatCryptoError (same boundary normalization)."""
    priv, pub = generate_x25519_keypair()
    ct_b64, envelopes = encrypt_blob_payload(
        plaintext=b"secret data",
        recipients=[RecipientKey("owner_user", "u-1", pub)],
    )
    env_json = json.loads(base64.b64decode(envelopes[0].encrypted_data_key_base64))
    wsk = base64.b64decode(env_json["wrappedSymmetricKeyBase64"])
    tampered_wsk = wsk[:-1] + bytes([wsk[-1] ^ 0xFF])
    env_json["wrappedSymmetricKeyBase64"] = base64.b64encode(tampered_wsk).decode()
    tampered_env_b64 = base64.b64encode(json.dumps(env_json).encode()).decode()

    with pytest.raises(VaultbeatCryptoError):
        decrypt_blob_payload(
            ciphertext_base64=ct_b64,
            encrypted_data_key_base64=tampered_env_b64,
            private_key_base64=priv,
        )


# ---------------------------------------------------------------------------
# Negative tests: malformed envelope JSON
# ---------------------------------------------------------------------------


def test_decrypt_blob_payload_rejects_non_json_envelope() -> None:
    """crypto.py line 114: raises VaultbeatCryptoError when envelope bytes are not valid JSON."""
    priv, pub = generate_x25519_keypair()
    ct_b64, _ = encrypt_blob_payload(
        plaintext=b"x",
        recipients=[RecipientKey("owner_user", "u-1", pub)],
    )
    not_json_env_b64 = base64.b64encode(b"this is not json at all").decode()

    with pytest.raises(VaultbeatCryptoError, match="encrypted_data_key is not a JSON envelope"):
        decrypt_blob_payload(
            ciphertext_base64=ct_b64,
            encrypted_data_key_base64=not_json_env_b64,
            private_key_base64=priv,
        )


def test_decrypt_blob_payload_rejects_json_array_envelope() -> None:
    """crypto.py line 117: raises VaultbeatCryptoError when envelope is a JSON array, not object."""
    priv, pub = generate_x25519_keypair()
    ct_b64, _ = encrypt_blob_payload(
        plaintext=b"x",
        recipients=[RecipientKey("owner_user", "u-1", pub)],
    )
    array_env_b64 = base64.b64encode(json.dumps([1, 2, 3]).encode()).decode()

    with pytest.raises(VaultbeatCryptoError, match="encrypted_data_key envelope must be an object"):
        decrypt_blob_payload(
            ciphertext_base64=ct_b64,
            encrypted_data_key_base64=array_env_b64,
            private_key_base64=priv,
        )


@pytest.mark.parametrize(
    "missing_field",
    ["senderPublicKeyBase64", "wrappedSymmetricKeyBase64"],
    ids=["missing-senderPubKey", "missing-wrappedSymKey"],
)
def test_decrypt_blob_payload_rejects_envelope_missing_required_field(missing_field: str) -> None:
    """crypto.py line 121-122: raises VaultbeatCryptoError when a required envelope field is absent."""
    priv, pub = generate_x25519_keypair()
    ct_b64, envelopes = encrypt_blob_payload(
        plaintext=b"x",
        recipients=[RecipientKey("owner_user", "u-1", pub)],
    )
    env_json = json.loads(base64.b64decode(envelopes[0].encrypted_data_key_base64))
    del env_json[missing_field]
    tampered_env_b64 = base64.b64encode(json.dumps(env_json).encode()).decode()

    with pytest.raises(VaultbeatCryptoError, match="encrypted_data_key envelope is missing key material"):
        decrypt_blob_payload(
            ciphertext_base64=ct_b64,
            encrypted_data_key_base64=tampered_env_b64,
            private_key_base64=priv,
        )


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("senderPublicKeyBase64", "not!!valid!!base64"),
        ("wrappedSymmetricKeyBase64", "also!!bad"),
    ],
    ids=["bad-senderPubKey-base64", "bad-wrappedSymKey-base64"],
)
def test_decrypt_blob_payload_rejects_non_base64_envelope_fields(field: str, bad_value: str) -> None:
    """_b64decode raises VaultbeatCryptoError when an envelope field is not valid base64."""
    priv, pub = generate_x25519_keypair()
    ct_b64, envelopes = encrypt_blob_payload(
        plaintext=b"x",
        recipients=[RecipientKey("owner_user", "u-1", pub)],
    )
    env_json = json.loads(base64.b64decode(envelopes[0].encrypted_data_key_base64))
    env_json[field] = bad_value
    tampered_env_b64 = base64.b64encode(json.dumps(env_json).encode()).decode()

    with pytest.raises(VaultbeatCryptoError, match="invalid base64 field"):
        decrypt_blob_payload(
            ciphertext_base64=ct_b64,
            encrypted_data_key_base64=tampered_env_b64,
            private_key_base64=priv,
        )


def test_decrypt_blob_payload_rejects_non_base64_ciphertext() -> None:
    """_b64decode raises VaultbeatCryptoError when ciphertext_base64 is not valid base64."""
    priv, pub = generate_x25519_keypair()
    _, envelopes = encrypt_blob_payload(
        plaintext=b"x",
        recipients=[RecipientKey("owner_user", "u-1", pub)],
    )
    with pytest.raises(VaultbeatCryptoError, match="invalid base64 field: ciphertext"):
        decrypt_blob_payload(
            ciphertext_base64="not!!valid!!base64",
            encrypted_data_key_base64=envelopes[0].encrypted_data_key_base64,
            private_key_base64=priv,
        )


def test_decrypt_blob_payload_rejects_non_base64_private_key() -> None:
    """_b64decode raises VaultbeatCryptoError when private_key_base64 is not valid base64."""
    priv, pub = generate_x25519_keypair()
    ct_b64, envelopes = encrypt_blob_payload(
        plaintext=b"x",
        recipients=[RecipientKey("owner_user", "u-1", pub)],
    )
    with pytest.raises(VaultbeatCryptoError, match="invalid base64 field: private_key_base64"):
        decrypt_blob_payload(
            ciphertext_base64=ct_b64,
            encrypted_data_key_base64=envelopes[0].encrypted_data_key_base64,
            private_key_base64="not!!valid!!base64",
        )


def test_decrypt_blob_payload_rejects_non_base64_encrypted_data_key() -> None:
    """_b64decode raises VaultbeatCryptoError when encrypted_data_key_base64 is not valid base64."""
    priv, pub = generate_x25519_keypair()
    ct_b64, _ = encrypt_blob_payload(
        plaintext=b"x",
        recipients=[RecipientKey("owner_user", "u-1", pub)],
    )
    with pytest.raises(VaultbeatCryptoError, match="invalid base64 field: encrypted_data_key"):
        decrypt_blob_payload(
            ciphertext_base64=ct_b64,
            encrypted_data_key_base64="not!!valid!!base64",
            private_key_base64=priv,
        )
