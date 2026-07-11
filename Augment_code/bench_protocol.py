"""Read-only protocol metadata for the frozen FITR benchmark."""

from __future__ import annotations

PRESERVED_ASSET_IDS: dict[str, list[str]] = {
    "Laptop": ["9604", "9682", "10223"],
    "Display": ["4523", "3386", "3393"],
    "Microwave": ["3456", "3458", "3461"],
    "Drawer": ["19179", "19203", "21467"],
    "Lamp": ["14563", "13928", "14372"],
    "Faucet": ["148", "988", "149"],
    "Knife": ["103", "395", "834"],
    "Dishwasher": ["11453", "11710", "11763", "11622"],
    "Door": ["8850", "9280", "8867"],
    "Refrigerator": ["9978", "9985", "10007"],
    "Scissors": ["10449", "10471", "10514"],
    "StorageFurniture": ["35059", "39030", "41002"],
}
