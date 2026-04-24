import os
import subprocess
from pathlib import Path

def upload_secrets():
    env_file = Path(".env")
    if not env_file.exists():
        print(".env file not found")
        return

    with open(env_file, "r") as f:
        lines = f.readlines()

    current_key = None
    current_value = []
    in_multiline = False

    for line in lines:
        line = line.strip()
        
        # Skip comments and empty lines if not in multiline
        if not in_multiline:
            if not line or line.startswith("#"):
                continue
            
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                
                # Check for multiline start
                if value.startswith('"') and not value.endswith('"'):
                    in_multiline = True
                    current_key = key
                    current_value = [value[1:]] # Remove starting quote
                elif value.startswith('"') and value.endswith('"'):
                    # Single line quoted
                    set_secret(key, value[1:-1])
                else:
                    # Single line unquoted
                    set_secret(key, value)
        else:
            # In multiline
            if line.endswith('"'):
                in_multiline = False
                current_value.append(line[:-1]) # Remove ending quote
                set_secret(current_key, "\n".join(current_value).replace("\\n", "\n"))
                current_key = None
                current_value = []
            else:
                current_value.append(line)

def set_secret(key, value):
    if not value:
        print(f"Skipping empty value for {key}")
        return
        
    # Handle escaped newlines
    value = value.replace("\\n", "\n")
    
    print(f"Setting secret: {key}", flush=True)
    try:
        # Use subprocess.run with input to handle multiline values correctly
        process = subprocess.run(
            ["gh", "secret", "set", key],
            input=value.encode('utf-8'),
            check=True,
            capture_output=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error setting secret {key}: {e.stderr.decode()}")

if __name__ == "__main__":
    upload_secrets()
