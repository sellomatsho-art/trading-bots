"""Structural guard: this project must stay incapable of touching funds.

The npm/PyPI weather-bot ecosystem is full of projects that ask for a hot
wallet key in a plaintext .env, and at least one shipped a dependency that was
later pulled from the registry for stealing exactly that. The defence here is
not a warning in the README -- it is that the code has no way to sign anything.
These tests fail the build if that ever stops being true.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "weatherbot"
SOURCES = sorted(PACKAGE.glob("*.py")) + [ROOT / "app.py"]

FORBIDDEN_IMPORTS = {
    "web3",
    "eth_account",
    "eth_keys",
    "eth_utils",
    "py_clob_client",
    "ethers",
    "hdwallet",
    "mnemonic",
    "bip_utils",
    "solders",
    "solana",
}

FORBIDDEN_TOKENS = (
    "private_key",
    "privatekey",
    "mnemonic",
    "seed_phrase",
    "seedphrase",
    "secret_key",
    "signer",
    "sign_order",
    "place_order",
    "create_order",
    "api_secret",
    "passphrase",
)


def _parsed():
    for path in SOURCES:
        if path.exists():
            yield path, ast.parse(path.read_text(), filename=str(path))


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Ids of Constant nodes that are docstrings, so prose can discuss keys."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    out.add(id(body[0].value))
    return out


def test_sources_are_present():
    assert len(SOURCES) >= 8, "guard would pass vacuously if it found no files"


def test_no_wallet_or_signing_libraries_are_imported():
    for path, tree in _parsed():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            banned = set(names) & FORBIDDEN_IMPORTS
            assert not banned, f"{path.name} imports {banned}"


def test_no_key_shaped_identifiers_or_literals():
    for path, tree in _parsed():
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                text = node.id.lower()
            elif isinstance(node, ast.Attribute):
                text = node.attr.lower()
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                text = node.name.lower()
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) in docstrings:
                    continue
                text = node.value.lower()
            else:
                continue
            hits = [t for t in FORBIDDEN_TOKENS if t in text]
            assert not hits, f"{path.name} references {hits} outside a docstring"


def test_the_package_never_reads_the_environment():
    """No env access at all means no .env full of keys can even be consumed."""
    for path, tree in _parsed():
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"environ", "getenv"}:
                pytest.fail(f"{path.name} reads the environment")


def test_all_http_calls_are_reads():
    """Only GET. A bot that cannot POST cannot submit an order."""
    allowed = {"get"}
    for path, tree in _parsed():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            method = node.func.attr
            if method in {"post", "put", "patch", "delete"}:
                pytest.fail(f"{path.name} performs a {method.upper()}")
            if method in allowed:
                continue


def test_requirements_pull_in_no_trading_or_wallet_packages():
    requirements = (ROOT / "requirements.txt").read_text().lower()
    for banned in ("web3", "eth-account", "py-clob-client", "clob", "solana", "mnemonic"):
        assert banned not in requirements, f"requirements.txt contains {banned}"


def test_simulation_only_flag_is_set():
    import weatherbot

    assert weatherbot.SIMULATION_ONLY is True
