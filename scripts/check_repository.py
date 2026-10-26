#!/usr/bin/env python
"""Check Git publishable files for credentials, local paths, size and syntax."""
import ast
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(
    ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT
).decode().split("\0")
machine_paths = re.compile(r"/(?:home|root|mnt|media|srv|opt|[A-Za-z]+-nas)/")
secret_patterns = {
    "access token": re.compile(rb"(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9_-]{20,})"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "credentials in URL": re.compile(rb"https?://[^\s/:]+:[^\s/@]+@"),
}
local_directories = {".aws", ".ssh", ".codex", ".agents", ".cache", ".hf-downloads", "__pycache__"}
errors = []
for relative in sorted(set(filter(None, paths))):
    path = ROOT / relative
    if path.is_symlink() and not path.resolve().is_relative_to(ROOT):
        errors.append(f"{relative}: symlink points outside the repository")
        continue
    if (set(path.relative_to(ROOT).parts) & local_directories
            or path.name in {".netrc", ".pypirc", ".env"}
            or (path.name.startswith(".env.") and path.name != ".env.example")
            or path.suffix in {".pem", ".key", ".orig", ".rej", ".bak", ".swp", ".swo"}):
        errors.append(f"{relative}: local credential or temporary file")
    if not path.is_file():
        continue
    if path.stat().st_size > 50 * 1024 * 1024:
        errors.append(f"{relative}: file exceeds 50 MiB")
    data = path.read_bytes()
    for label, pattern in secret_patterns.items():
        if pattern.search(data):
            errors.append(f"{relative}: possible {label}; inspect locally")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        continue
    if path.resolve() != Path(__file__).resolve():
        if machine_paths.search(text):
            errors.append(f"{relative}: contains a machine-specific path")
    if path.suffix == ".py":
        try:
            ast.parse(text, filename=relative)
        except SyntaxError as exc:
            errors.append(str(exc))
if errors:
    print("\n".join(errors), file=sys.stderr)
    raise SystemExit(1)
print(f"Checked {len(set(filter(None, paths)))} source files.")
