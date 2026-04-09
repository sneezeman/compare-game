"""
WSGI entry point for gunicorn.

In production, gunicorn binds to 127.0.0.1:8001 (plain HTTP) and Caddy
in front terminates TLS on :443 and reverse-proxies to us. Direct TLS
termination in gunicorn is no longer used because gunicorn's gthread
worker has no handshake timeout, making it vulnerable to slow-loris
TLS attacks that can wedge all worker threads.

Usage:
    gunicorn wsgi:app --bind 127.0.0.1:8001 --workers 1 --threads 4 --timeout 300

Required environment variables:
    COMPARE_DATA_DIR          Path to directory containing GIF files
    COMPARE_USERS_CONFIG      Path to users.json (for basic auth)
"""

import os
import sys

from werkzeug.middleware.proxy_fix import ProxyFix

# Import everything from the app module.
# NOTE: both 'from app import ...' and 'import app as app_module' reference
# the SAME module object in sys.modules['app']. There is no dual-module
# issue — globals modified by load_users_config() / load_data_dir() are
# visible to check_auth() and the route handlers.
from app import app, load_data_dir, scan_past_results, load_users_config
import app as app_module

# ---------------------------------------------------------------------------
# Initialization — runs once when the gunicorn worker imports this module
# ---------------------------------------------------------------------------

data_dir = os.environ.get('COMPARE_DATA_DIR', '/opt/compare-game/data/ls3231/')
users_cfg = os.environ.get('COMPARE_USERS_CONFIG', '')

print(f'[wsgi] COMPARE_DATA_DIR  = {data_dir!r}', file=sys.stderr)
print(f'[wsgi] COMPARE_USERS_CONFIG = {users_cfg!r}', file=sys.stderr)
print(f'[wsgi] data dir exists?  {os.path.isdir(data_dir)}', file=sys.stderr)
print(f'[wsgi] users.json exists? {os.path.isfile(users_cfg) if users_cfg else "NOT SET"}',
      file=sys.stderr)

# Set results directory
app_module.results_dir = os.environ.get('COMPARE_RESULTS_DIR',
                                        '/opt/compare-game/results')

# Load users — reads COMPARE_USERS_CONFIG env var internally
load_users_config()
print(f'[wsgi] _user_config has {len(app_module._user_config)} user(s)', file=sys.stderr)
if not app_module._user_config:
    print('[wsgi] WARNING: _user_config is empty — auth will be DISABLED!', file=sys.stderr)

# Load GIF data
load_data_dir(data_dir)
print(f'[wsgi] experiments has {len(app_module.experiments)} GIF(s)', file=sys.stderr)
if not app_module.experiments:
    print('[wsgi] WARNING: no GIFs found — the UI will be empty!', file=sys.stderr)

# Scan past results
scan_past_results()

# Caddy on the same host terminates TLS and forwards to us. Trust exactly
# one upstream proxy hop so request.remote_addr reflects the real client IP
# (otherwise the rate limiter / ban list would only ever see 127.0.0.1 and
# could lock everyone out after one bad actor).
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
print('[wsgi] ProxyFix installed (x_for=1, x_proto=1, x_host=1)', file=sys.stderr)
