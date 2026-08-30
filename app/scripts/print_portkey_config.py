"""Print Portkey Config JSON to paste into the Portkey dashboard.

Usage:
  uv run python -m app.scripts.print_portkey_config

Then: Portkey → Configs → Create → paste JSON → copy the ``pc-...`` ID into
``.env`` as ``PORTKEY_CONFIG_ID=pc-...``.
"""

from __future__ import annotations

from app.gateway.portkey_client import describe_inline_config, gateway_status


def main() -> None:
    status = gateway_status()
    print("=== Current Portkey status ===")
    for k, v in status.items():
        print(f"  {k}: {v}")
    print()
    print("=== Paste this JSON into Portkey -> Configs ===")
    print(describe_inline_config())
    print()
    print("Then set in .env:")
    print("  PORTKEY_CONFIG_ID=pc-xxxxxxxx")
    print("  PORTKEY_ALLOW_INLINE_CONFIG=false")


if __name__ == "__main__":
    main()
