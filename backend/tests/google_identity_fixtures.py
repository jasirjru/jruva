"""Synthetic signing keys; actual Google-auth RSA and claim verification stays real."""
import json
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from app.api.v1.endpoints import auth
from app.services.auth_service import auth_service
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from google.auth import crypt, jwt
from google.auth.transport import requests as google_requests

CLIENT_ID = "synthetic-client.apps.googleusercontent.com"

@pytest.fixture(scope="module")
def signing_keys():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Synthetic Google test key")])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=1))
            .sign(key, hashes.SHA256()))
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    return crypt.RSASigner.from_string(pem, key_id="synthetic-key"), cert.public_bytes(serialization.Encoding.PEM).decode()


@pytest.fixture
def google_token_factory(monkeypatch, signing_keys, tmp_path):
    signer, certificate = signing_keys
    class CertResponse:
        status = 200
        data = json.dumps({"synthetic-key": certificate}).encode()
    def request(*args, **kwargs):
        return CertResponse()
    monkeypatch.setattr(google_requests, "Request", lambda **kwargs: request)
    monkeypatch.setattr(auth_service, "db_path", str(tmp_path / "google-auth.db"))
    auth_service._init_db()
    monkeypatch.setattr(auth, "settings", SimpleNamespace(ENVIRONMENT="test", GOOGLE_CLIENT_ID=CLIENT_ID))
    def signed(**changes):
        claims = {"iss": "https://accounts.google.com", "aud": CLIENT_ID,
                  "iat": int(time.time()) - 1, "exp": int(time.time()) + 120,
                  "sub": "synthetic-google-sub", "email": "synthetic-oidc@example.invalid",
                  "email_verified": True, "name": "Synthetic OIDC"}
        claims.update(changes)
        return jwt.encode(signer, claims).decode()
    return signed
