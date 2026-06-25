from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import os
from google_auth_oauthlib.flow import InstalledAppFlow, WSGITimeoutError
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.readonly",
]

ROOT_DIR = Path(__file__).resolve().parents[1]
TOKEN_PATH = ROOT_DIR / "token.json"
CREDENTIALS_PATH = ROOT_DIR / "credentials.json"


def get_gmail_service() -> Any:
    """Return an authenticated Gmail API service object.

    This utility checks for an existing token.json file and attempts to reuse
    it if valid. If the token is missing, expired, or otherwise invalid, it
    runs the installed app flow using credentials.json and saves a new token.
    """
    creds: Credentials | None = None

    try:
        if TOKEN_PATH.exists():
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not CREDENTIALS_PATH.exists():
                    raise FileNotFoundError(
                        f"Missing credentials file: {CREDENTIALS_PATH}"
                    )

                flow = InstalledAppFlow.from_client_secrets_file(
                    str(CREDENTIALS_PATH), SCOPES
                )

                creds = _run_flow_with_retries(flow)

            _save_token(creds)

        return build("gmail", "v1", credentials=creds)

    except FileNotFoundError as exc:
        raise RuntimeError(
            "Gmail authentication failed because a required file was not found. "
            f"Ensure '{CREDENTIALS_PATH.name}' exists in the repository root."
        ) from exc
    except (GoogleAuthError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Gmail authentication failed while loading or refreshing credentials. "
            "Delete 'token.json' and retry if the token file is corrupted."
        ) from exc
    except Exception as exc:
        raise RuntimeError("Unexpected error during Gmail authentication.") from exc


def _save_token(credentials: Credentials) -> None:
    try:
        with TOKEN_PATH.open("w", encoding="utf-8") as token_file:
            token_file.write(credentials.to_json())
    except OSError as exc:
        raise RuntimeError(
            f"Unable to save Gmail token to '{TOKEN_PATH}'."
        ) from exc


def _run_flow_with_retries(flow: InstalledAppFlow) -> Credentials:
    """Run the OAuth2 local server flow with retries and helpful error messages.

    Tries a sequence of ports (including a dynamic port) and increases the
    timeout. If running in a headless environment, it will print the auth URL
    for manual copy/paste.
    """
    host = os.getenv("GMAIL_OAUTH_HOST", "localhost")
    timeout = int(os.getenv("GMAIL_AUTH_TIMEOUT", "300"))

    ports = []
    env_port = os.getenv("GMAIL_OAUTH_PORT")
    if env_port:
        try:
            ports.append(int(env_port))
        except ValueError:
            pass

    # Try a common fixed port first, then let the OS pick a free port (0)
    ports.extend([8080, 0])

    last_exc: Exception | None = None
    open_browser = True

    for port in ports:
        try:
            return flow.run_local_server(
                host=host, port=port, timeout_seconds=timeout, open_browser=open_browser
            )
        except WSGITimeoutError as exc:
            last_exc = exc
            if open_browser:
                try:
                    auth_url, _ = flow.authorization_url()
                except Exception:
                    auth_url = None

                print(
                    "Timeout waiting for OAuth callback. If your environment is headless,"
                    " open the URL below in a browser manually."
                )
                if auth_url:
                    print(auth_url)

                # Retry but don't attempt to open the browser this time
                open_browser = False
                try:
                    return flow.run_local_server(
                        host=host, port=port, timeout_seconds=timeout, open_browser=open_browser
                    )
                except WSGITimeoutError as exc2:
                    last_exc = exc2
                    continue
        except Exception as exc:
            last_exc = exc
            continue

    raise RuntimeError(
        "Gmail OAuth local server timed out waiting for the authorization response. "
        "Try setting environment variable `GMAIL_OAUTH_PORT`, increasing `GMAIL_AUTH_TIMEOUT`, "
        "or running in an environment with a browser available."
    ) from last_exc
