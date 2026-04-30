import requests
import sys

# ============================
# CONFIG — EDIT ONLY THIS
# ============================
BASE_URL = "http://localhost:8000"
ADMIN_EMAIL = "admin@gmail.com"
ADMIN_PASSWORD = "123456"

# ============================
# CATEGORY DATA
# ============================
CATEGORIES = [
    "Fog / Aux Lights",
    "Headlamp Bulbs",
    "Tail Lamps",
    "Indicators",
    "Flashers / DRLs / Strobe",
    "Wiring Harness",
    "Switches",
    "Relays",
    "Helmets",
    "Riding Jackets",
    "Riding Pants",
    "Knee Guards",
    "Crash Guards / Engine Guards",
    "Knuckle Guards",
    "Top Box",
    "Side Panniers",
    "Mounting Brackets / Clamps",
    "Bungee Cord",
    "Balaclava",
    "Helmet Chin Mount",
    "GPS Device",
    "Comms System",
    "Air Filters",
    "Exhaust Systems",
    "Brake Pads",
    "Spark Plug",
    "Engine Oil",
    "Chain Cleaners",
    "Side Stand Extenders",
    "Handle Grips",
    "Handle Bar Risers",
    "Key Chain",
    "Guardian Bell",
    "Helmet Light",
    "Helmet Lock",
    "7-Inch Headlight",
    "Bottle Holder",
    "Brake Oil",
    "Mobile Charger",
]

# ============================
# LOGIN
# ============================
def login():
    url = f"{BASE_URL}/api/auth/login"
    payload = {
        "email": ADMIN_EMAIL,
        "password": ADMIN_PASSWORD,
    }

    r = requests.post(url, json=payload)
    if r.status_code != 200:
        print("❌ Login failed:", r.text)
        sys.exit(1)

    token = r.json().get("access_token")
    if not token:
        print("❌ Token not received")
        sys.exit(1)

    print("✅ Logged in successfully")
    return token

# ============================
# CREATE CATEGORY
# ============================
def create_category(token, name):
    url = f"{BASE_URL}/api/categories"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "name": name
    }

    r = requests.post(url, json=payload, headers=headers)

    if r.status_code == 200 or r.status_code == 201:
        print(f"✅ Created: {name}")
    elif r.status_code == 400:
        print(f"⚠️ Skipped (already exists?): {name}")
    else:
        print(f"❌ Failed [{name}]: {r.status_code} {r.text}")

# ============================
# MAIN
# ============================
def main():
    token = login()

    print("\n🚀 Seeding categories...\n")

    for name in CATEGORIES:
        create_category(token, name)

    print("\n🎉 Category seeding completed")

if __name__ == "__main__":
    main()
