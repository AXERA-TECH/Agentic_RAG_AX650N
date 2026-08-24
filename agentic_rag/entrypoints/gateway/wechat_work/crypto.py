"""WeChat Work message encryption/decryption (AES-256-CBC + SHA1).

Reference: https://developer.work.weixin.qq.com/document/path/90968
"""

from __future__ import annotations

import base64
import hashlib
import random
import struct
import time


class WeChatCrypto:
    """Handle WeChat Work callback message encryption and signature verification.

    The EncodingAESKey is a 43-char Base64 string that decodes to a 32-byte AES key.
    Messages use AES-256-CBC with the AES key as IV (first 16 bytes of key).
    """

    BLOCK_SIZE = 32

    def __init__(self, token: str, encoding_aes_key: str, corp_id: str) -> None:
        self.token = token
        self.corp_id = corp_id
        # EncodingAESKey is 43-char Base64 → 32-byte AES key (add padding '=' for decoding)
        self.aes_key = base64.b64decode(encoding_aes_key + "=")

    # ── Signature Verification ─────────────────────────────────

    def verify_signature(
        self, msg_signature: str, timestamp: str, nonce: str, echostr: str
    ) -> bool:
        """Verify callback signature from WeChat Work server."""
        expected = self._sha1(timestamp, nonce, echostr)
        return msg_signature == expected

    # ── Message Decryption ─────────────────────────────────────

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt an encrypted message XML from WeChat Work.

        Returns the plain text XML string.
        """
        raw = base64.b64decode(ciphertext)
        plain = self._aes_decrypt(raw)

        # Strip PKCS#7 padding
        pad = plain[-1]
        plain = plain[:-pad]

        # Parse: random(16) + msg_len(4, big-endian) + msg + corp_id
        msg_len = struct.unpack("!I", plain[16:20])[0]
        msg = plain[20 : 20 + msg_len].decode("utf-8")
        received_corp_id = plain[20 + msg_len :].decode("utf-8")

        if received_corp_id != self.corp_id:
            raise ValueError(
                f"Corp ID mismatch: expected {self.corp_id!r}, got {received_corp_id!r}"
            )
        return msg

    # ── Message Encryption ─────────────────────────────────────

    def encrypt(self, plain_text: str) -> str:
        """Encrypt a reply message for WeChat Work.

        Returns base64-encoded ciphertext.
        """
        random_bytes = bytes(random.getrandbits(8) for _ in range(16))
        text_bytes = plain_text.encode("utf-8")
        corp_bytes = self.corp_id.encode("utf-8")
        msg_len = struct.pack("!I", len(text_bytes))

        raw = random_bytes + msg_len + text_bytes + corp_bytes

        # PKCS#7 padding to 32-byte block
        pad = self.BLOCK_SIZE - len(raw) % self.BLOCK_SIZE
        raw += bytes([pad] * pad)

        encrypted = self._aes_encrypt(raw)
        return base64.b64encode(encrypted).decode()

    # ── Signature Generation (for responses) ───────────────────

    def build_response_signature(self, encrypted_msg: str) -> str:
        """Build msg_signature for an encrypted response."""
        ts = str(int(time.time()))
        nonce = self._random_nonce()
        sig = self._sha1(ts, nonce, encrypted_msg)
        return sig, ts, nonce

    # ── Internal helpers ───────────────────────────────────────

    def _sha1(self, timestamp: str, nonce: str, msg: str) -> str:
        """SHA1 of sorted [token, timestamp, nonce, msg]."""
        params = sorted([self.token, timestamp, nonce, msg])
        return hashlib.sha1("".join(params).encode("utf-8")).hexdigest()

    def _aes_decrypt(self, data: bytes) -> bytes:
        from Crypto.Cipher import AES
        cipher = AES.new(self.aes_key, AES.MODE_CBC, iv=self.aes_key[:16])
        return cipher.decrypt(data)

    def _aes_encrypt(self, data: bytes) -> bytes:
        from Crypto.Cipher import AES
        cipher = AES.new(self.aes_key, AES.MODE_CBC, iv=self.aes_key[:16])
        return cipher.encrypt(data)

    @staticmethod
    def _random_nonce() -> str:
        return "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=16))
