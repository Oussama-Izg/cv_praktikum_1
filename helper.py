import cv2
import numpy as np


def find_inbus_contour(contours, min_area=500):
    candidates = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        rect = cv2.minAreaRect(c)
        _, (w, h), _ = rect
        if min(w, h) == 0:
            continue

        aspect = max(w, h) / min(w, h)

        # Solidity = Konturfläche / ConvexHull-Fläche
        # Inbus:   solidity < 0.8  (L-Form = nicht konvex)
        # Kreis:   solidity ≈ 1.0  (sehr konvex)
        # Rauschen: solidity ≈ 0.5 (unregelmäßig)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # Extent = Konturfläche / BoundingRect-Fläche
        # Inbus:  extent klein (viel leerer Raum im Rechteck)
        # Kreis:  extent ≈ 0.785
        x, y, bw, bh = cv2.boundingRect(c)
        extent = area / (bw * bh) if (bw * bh) > 0 else 0

        candidates.append({
            "contour": c,
            "area": area,
            "aspect": aspect,
            "solidity": solidity,
            "extent": extent,
        })

        print(f"  Fläche={int(area):6d}  "
              f"Aspect={aspect:5.1f}  "
              f"Solidity={solidity:.2f}  "
              f"Extent={extent:.2f}")

    # ── Filter 1: Aspect > 2 → muss länglich sein ──
    candidates = [c for c in candidates if c["aspect"] > 2]

    # ── Filter 2: Solidity < 0.85 → L-Form, nicht Kreis ──
    candidates = [c for c in candidates if c["solidity"] < 0.85]

    # ── Filter 3: Extent < 0.5 → viel Leerraum im Rechteck ──
    candidates = [c for c in candidates if c["extent"] < 0.5]

    if not candidates:
        print("Warnung: Kein Kandidat nach Filterung – Filter lockern")
        return None

    # ── Bester Kandidat: größte Fläche unter den Gefilterten ──
    best = max(candidates, key=lambda c: c["area"])
    return best["contour"]