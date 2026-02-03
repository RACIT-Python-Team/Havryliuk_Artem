ZONES = {
    "Bedroom":        (24, 152, 206, 372),
    "Bathroom 1":     (24, 27, 143, 142),
    "Bathroom 2":     (497, 26, 615, 106),
    "Pantry 1":       (24, 392, 190, 472),
    "Pantry 2":       (449, 392, 615, 472),
    "Hall":           (154, 26, 486, 144),
    "Master Bedroom": (216, 152, 426, 450),
    "Kitchen":        (433, 151, 617, 372)
}

def active_zone(x, y):
    for zone_name, (x1, y1, x2, y2) in ZONES.items():
        if x1 < x < x2 and y1 < y < y2:
            return zone_name
    return None