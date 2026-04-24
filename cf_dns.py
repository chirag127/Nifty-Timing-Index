import requests
import json
import os

CF_API_KEY = "99fe5f342556fb3a1585eefe7bcdb93af0c2c"
CF_EMAIL = "whyiswhen@gmail.com"

headers = {
    "X-Auth-Email": CF_EMAIL,
    "X-Auth-Key": CF_API_KEY,
    "Content-Type": "application/json"
}

# 1. Get Zone ID for oriz.in
res = requests.get("https://api.cloudflare.com/client/v4/zones?name=oriz.in", headers=headers)
print("Zones Response:", res.status_code, res.text)
data = res.json()

if data["success"] and len(data["result"]) > 0:
    zone_id = data["result"][0]["id"]
    print(f"Found Zone ID for oriz.in: {zone_id}")
    
    # 2. Add CNAME record
    dns_record = {
        "type": "CNAME",
        "name": "nti",
        "content": "nifty-timing-index.pages.dev",
        "ttl": 1, # Auto
        "proxied": True
    }
    
    res = requests.post(f"https://api.cloudflare.com/client/v4/zones/{zone_id}/dns_records", headers=headers, json=dns_record)
    print("DNS Record Add Response:", res.status_code, res.text)
else:
    print("Could not find zone for oriz.in")
