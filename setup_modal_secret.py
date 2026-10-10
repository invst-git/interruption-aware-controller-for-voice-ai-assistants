"""setup_modal_secret.py -- one-time setup: creates the Modal secret
that modal_controller.py's predict endpoint checks against, and prints
the .env lines to add locally.

Run once (needs `modal token set` / `modal setup` already done):
    python setup_modal_secret.py

Safe to re-run: pass --force to rotate the key (invalidates the old
one -- update .env on every machine that calls the endpoint after
rotating).
"""
import secrets
import subprocess
import sys

SECRET_NAME = "candor-fd-api-key"


def main():
    force = "--force" in sys.argv
    token = secrets.token_urlsafe(32)

    cmd = ["modal", "secret", "create", SECRET_NAME, f"API_KEY={token}"]
    if force:
        cmd.append("--force")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        if "already exists" in (result.stderr + result.stdout) and not force:
            print("\nSecret already exists. Re-run with --force to rotate it "
                  "(this invalidates the old key -- update .env everywhere after).")
        sys.exit(1)

    print(f"Created Modal secret '{SECRET_NAME}'.\n")
    print("Add this to your local .env (do not commit it):")
    print(f"MODAL_CONTROLLER_API_KEY={token}")
    print("\nMODAL_CONTROLLER_URL comes from `modal deploy modal_controller.py`'s "
          "printed endpoint URL -- add that to .env too.")


if __name__ == "__main__":
    main()
