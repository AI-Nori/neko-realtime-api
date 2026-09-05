"""Tests for OpenAI-compatible transcription endpoint (POST /v1/audio/transcriptions).

Verifies:
- 路由已注册到 FastAPI app
- 认证开启时无 Bearer Token 返回 401
- 缺少音频文件返回 422（FastAPI multipart 校验）
- 本地与远程都不可用时返回 503
"""
from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.config import ServerConfig
from server.transcription_endpoint import register_transcription_routes


def _make_silence_wav(seconds: float = 0.5, sample_rate: int = 16000) -> bytes:
    """生成一段指定时长的静音 WAV 字节流，供 multipart 上传测试使用。"""
    n = int(seconds * sample_rate)
    audio_int16 = (np.random.randn(n) * 100).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


@pytest.fixture()
def reset_config():
    """每个测试前后重置 ServerConfig 单例。"""
    ServerConfig.reset()
    yield
    ServerConfig.reset()


def _install_config(monkeypatch, **overrides) -> None:
    """覆盖 ServerConfig.load() 以返回测试用配置。

    必须在 register_transcription_routes() 之前调用，因为路由注册时
    会一次性读取配置并闭包到 endpoint 内部。
    """
    cfg = ServerConfig()  # 默认配置
    # 套上指定覆盖项
    for path, value in overrides.items():
        keys = path.split(".")
        node = cfg._data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    monkeypatch.setattr(ServerConfig, "load", classmethod(lambda cls, p=None: cfg))


def _make_app(**config_overrides) -> FastAPI:
    """先安装配置，再注册路由 - 保证路由闭包到测试配置。"""
    app = FastAPI()
    register_transcription_routes(app)
    return app


def test_route_registered() -> None:
    """路由 POST /v1/audio/transcriptions 必须存在。"""
    ServerConfig.reset()
    app = FastAPI()
    register_transcription_routes(app)
    paths = {(route.path, frozenset(route.methods or [])) for route in app.routes}
    assert ("/v1/audio/transcriptions", frozenset({"POST"})) in paths
    ServerConfig.reset()


def test_missing_bearer_returns_401(reset_config, monkeypatch) -> None:
    """认证开启 + 无 Bearer 头 → 401。"""
    _install_config(
        monkeypatch,
        **{"security.auth_enabled": True, "security.auth_token": "secret-token"},
    )
    app = _make_app()
    client = TestClient(app)
    wav_bytes = _make_silence_wav()

    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == 401, resp.text


def test_wrong_bearer_returns_401(reset_config, monkeypatch) -> None:
    """认证开启 + 错误 Bearer → 401。"""
    _install_config(
        monkeypatch,
        **{"security.auth_enabled": True, "security.auth_token": "secret-token"},
    )
    app = _make_app()
    client = TestClient(app)
    wav_bytes = _make_silence_wav()

    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401, resp.text


def test_no_backend_returns_503(reset_config, monkeypatch) -> None:
    """认证关闭 + local_asr=false + 无远程 base_url → 503。"""
    _install_config(
        monkeypatch,
        **{
            "security.auth_enabled": False,
            "realtime_server.auth_enabled": False,
            "services.asr.local_asr": False,
            "services.asr.base_url": None,
        },
    )
    app = _make_app()
    client = TestClient(app)
    wav_bytes = _make_silence_wav()

    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == 503, resp.text
    assert "ASR backend unavailable" in resp.json().get("detail", "")


def test_empty_payload_returns_400(reset_config, monkeypatch) -> None:
    """空文件 → 400。"""
    _install_config(
        monkeypatch,
        **{"security.auth_enabled": False, "realtime_server.auth_enabled": False},
    )
    app = _make_app()
    client = TestClient(app)

    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", b"", "audio/wav")},
    )
    assert resp.status_code == 400, resp.text


def test_oversize_upload_returns_413(reset_config, monkeypatch) -> None:
    """超过 security.max_upload_bytes 的上传 → 413（防超大上传耗尽内存）。

    上传体小于 Content-Length 预检余量（1MB），因此本用例走的是
    分块读取路径中的实际字节计数限制，而非头部预检。
    """
    _install_config(
        monkeypatch,
        **{
            "security.auth_enabled": False,
            "realtime_server.auth_enabled": False,
            "services.asr.local_asr": False,
            "services.asr.base_url": None,
            "security.max_upload_bytes": 1024,
        },
    )
    app = _make_app()
    client = TestClient(app)
    wav_bytes = _make_silence_wav()  # ~16KB，远超 1024 上限

    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == 413, resp.text


def test_oversize_rejected_before_auth_and_parsing(reset_config, monkeypatch) -> None:
    """超大请求体必须在认证与 multipart 解析之前被拒绝 → 413。

    开启认证且不携带 Bearer：若超大检查仍发生在 endpoint 内部
    （认证之后、multipart 解析之后），本用例会先得到 401；修复后
    HTTP middleware 读取 Content-Length 即拒绝，直接返回 413。
    """
    _install_config(
        monkeypatch,
        **{
            "security.auth_enabled": True,
            "realtime_server.auth_enabled": True,
            "security.auth_token": "secret-token",
            "services.asr.local_asr": False,
            "services.asr.base_url": None,
            "security.max_upload_bytes": 1024,
        },
    )
    app = _make_app()
    client = TestClient(app)
    # 1MB+1KB+1 字节，超过 1024 上限 + 1MB Content-Length 预检余量
    big_payload = b"x" * (1024 + 1024 * 1024 + 1)

    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", big_payload, "audio/wav")},
        # 故意不携带 Authorization 头
    )
    assert resp.status_code == 413, resp.text
    assert "Upload exceeds maximum allowed size" in resp.json().get("detail", "")
