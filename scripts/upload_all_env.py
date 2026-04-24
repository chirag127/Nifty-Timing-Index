import os
import subprocess
import re
from pathlib import Path

# Keywords that indicate a sensitive secret
SECRET_KEYWORDS = [
    "KEY", "SECRET", "PASSWORD", "TOKEN", "PRIVATE", "AUTH", "CREDENTIAL", 
    "BASE64", "API", "DSN", "URL", "ADDRESS", "EMAIL"
]

# Exceptions: some things might have these keywords but are not really secrets
# or are already used as secrets in workflows and we don't want to break them yet.
# Actually, it's safer to move as many as possible to variables.

def is_secret(key, value):
    # If it's already a secret in GitHub, keep it as a secret if possible
    # but we need to free up space.
    
    # Check keywords
    for kw in SECRET_KEYWORDS:
        if kw in key.upper():
            return True
    
    # Check value for suspicious patterns (like long hex or base64)
    if len(value) > 100:
        return True
        
    return False

def upload_all():
    env_file = Path(".env")
    if not env_file.exists():
        print(".env not found")
        return

    # Parse .env
    env_vars = {}
    with open(env_file, "r") as f:
        lines = f.readlines()
    
    current_key = None
    current_value = []
    in_multiline = False
    
    for line in lines:
        raw_line = line
        line = line.strip()
        if not in_multiline:
            if not line or line.startswith("#"): continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if value.startswith('"') and not value.endswith('"'):
                    in_multiline = True
                    current_key = key
                    current_value = [value[1:]]
                elif value.startswith('"') and value.endswith('"'):
                    env_vars[key] = value[1:-1].replace("\\n", "\n")
                else:
                    env_vars[key] = value
        else:
            if line.endswith('"'):
                in_multiline = False
                current_value.append(line[:-1])
                env_vars[current_key] = "\n".join(current_value).replace("\\n", "\n")
            else:
                current_value.append(line)

    # Categorize
    secrets = {}
    variables = {}
    
    for k, v in env_vars.items():
        if k.startswith("GITHUB_"): continue # Skip forbidden prefix
        if is_secret(k, v):
            secrets[k] = v
        else:
            variables[k] = v

    print(f"Total variables: {len(env_vars)}")
    print(f"Categorized: {len(secrets)} secrets, {len(variables)} variables")

    # Upload Variables
    for k, v in variables.items():
        print(f"Setting variable: {k}")
        subprocess.run(["gh", "variable", "set", k, "--body", v], check=False)
        # Delete secret if it exists to free up space
        subprocess.run(["gh", "secret", "delete", k], check=False)

    # Upload Secrets
    # We might still hit the limit if secrets > 100
    for k, v in secrets.items():
        print(f"Setting secret: {k}")
        process = subprocess.run(
            ["gh", "secret", "set", k],
            input=v.encode('utf-8'),
            check=False,
            capture_output=True
        )
        if process.returncode != 0:
            print(f"Failed to set secret {k}: {process.stderr.decode().strip()}")

    # Update Workflows
    update_workflows(variables.keys())

def update_workflows(var_names):
    workflow_dir = Path(".github/workflows")
    if not workflow_dir.exists(): return
    
    for wf in workflow_dir.glob("*.yml"):
        content = wf.read_text()
        new_content = content
        for var in var_names:
            # Replace ${{ secrets.VAR }} with ${{ vars.VAR }}
            pattern = r'\$\{\{\s*secrets\.' + re.escape(var) + r'\s*\}\}'
            replacement = f'${{{{ vars.{var} }}}}'
            new_content = re.sub(pattern, replacement, new_content)
        
        if new_content != content:
            print(f"Updating workflow: {wf.name}")
            wf.write_text(new_content)

if __name__ == "__main__":
    upload_all()
