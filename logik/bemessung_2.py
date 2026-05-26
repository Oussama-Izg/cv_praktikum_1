import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def measure_dimensions_by_minAreaRect(
    bild_pfad: str,
    muenz_durchmesser_mm: float = 22.25,
    ausgabe_ordner: str = "output"
):

    os.makedirs(ausgabe_ordner, exist_ok=True)

    # --- Bild laden ---
    img = cv2.imread(bild_pfad)
    if img is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {bild_pfad}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Vorverarbeitung ---
    grey    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel   = np.ones((5, 5), np.uint8)
    clean    = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean    = cv2.morphologyEx(clean,  cv2.MORPH_OPEN,  kernel, iterations=2)

    # --- Plot 1: Canny ---
    edges = cv2.Canny(clean, 50, 150)
    _save_side_by_side(
        img_rgb, "Original",
        edges,   "Canny Edge Detection",
        os.path.join(ausgabe_ordner, "01_canny_edges.png"),
        cmap_right="gray"
    )

    # --- Konturen ---
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Münze erkennen ---
    coin_contour, best_solidity = _finde_muenze(contours)
    if coin_contour is None:
        raise RuntimeError("Keine Münze gefunden!")

    # Da das Bild schräg ist, fitten wir eine Ellipse statt eines Kreises
    ellipse = cv2.fitEllipse(coin_contour)
    (cx, cy), (axes1, axes2), angle = ellipse

    # Die längste Achse (major axis) wird durch die Schräglage nicht gestaucht!
    d_major = max(axes1, axes2)
    d_minor = min(axes1, axes2)

    cx, cy = int(cx), int(cy)
    radius = int(d_major / 2)
    pixels_per_mm = d_major / muenz_durchmesser_mm

    # --- Plot 2: Münze erkennen ---
    coin_img = img_rgb.copy()
    cv2.ellipse(coin_img, ellipse, (0, 255, 0), 3)  # Ellipse zeichnen
    cv2.circle(coin_img, (cx, cy), 4, (255, 0, 0), -1)

    # Schiefstand (Verhältnis von kurzer zu langer Achse, 1.0 = perfekter Kreis)
    ratio = d_minor / d_major

    cv2.putText(coin_img,
                f"Solidity: {best_solidity:.2f} | Angle-Ratio: {ratio:.2f}",
                (cx - 100, cy - radius - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    _save_single(
        coin_img,
        f"Münze erkannt | Solidity: {best_solidity:.2f} | D_major: {d_major:.1f} px",
        os.path.join(ausgabe_ordner, "02_muenze_erkannt.png")
    )

    # Ab hier läuft der Code ganz normal mit "Inbus erkennen" weiter ...
    # (cx, cy und radius werden weiterhin für _finde_inbus verwendet)

    # --- Inbus erkennen ---
    inbus_contour = _finde_inbus(contours, cx, cy,
                                 radius)  # x, y zu cx, cy geändert
    if inbus_contour is None:
        raise RuntimeError("Kein Inbus gefunden!")

    # --- MinAreaRect + Maße ---
    rect = cv2.minAreaRect(inbus_contour)
    (_, _), (w, h), angle = rect
    if w > h:
        w, h = h, w
    box = np.int32(cv2.boxPoints(rect))
    width_mm = w / pixels_per_mm
    height_mm = h / pixels_per_mm

    # --- Plot 3: Ergebnis ---
    p1, p2, p3, _ = box
    long_mid = ((p2[0] + p3[0]) // 2, (p2[1] + p3[1]) // 2)
    short_mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

    final_img = img_rgb.copy()
    cv2.drawContours(final_img, [box], 0, (0, 255, 0), 3)
    # cv2.circle(final_img, (cx, cy), radius, (255, 165, 0), 2)  # x, y zu cx, cy geändert
    cv2.putText(final_img, f"{height_mm:.1f} mm", long_mid,
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.putText(final_img, f"{width_mm:.1f} mm", short_mid,
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    _save_single(
        final_img,
        f"MinAreaRect  |  Breite: {width_mm:.2f} mm  |  Länge: {height_mm:.2f} mm",
        os.path.join(ausgabe_ordner, "03_minAreaRect_ergebnis.png")
    )

    # --- Ergebnisse speichern ---
    ergebnisse = {
        "breite_mm": round(width_mm, 2),
        "laenge_mm": round(height_mm, 2),
        "pixels_per_mm": round(pixels_per_mm, 4),
        "solidity": round(best_solidity, 4),
        # best_circ zu best_solidity geändert
        "muenz_radius_px": radius,
    }
    _speichere_ergebnisse(ergebnisse, muenz_durchmesser_mm, ausgabe_ordner)

    print(f"Fertig! Ergebnisse in '{ausgabe_ordner}/' gespeichert.")
    print(f"  Breite:  {width_mm:.2f} mm")
    print(f"  Länge:   {height_mm:.2f} mm")
    return ergebnisse


# ============================================================
# Private Hilfsfunktionen
# ============================================================

def _finde_muenze(contours):
    best_coin_contour = None
    best_solidity = 0

    for c in contours:
        area = cv2.contourArea(c)
        # fitEllipse benötigt mindestens 5 Punkte
        if area < 200 or len(c) < 5:
            continue

        # Solidity = Konturfläche / Fläche der konvexen Hülle
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue

        solidity = area / hull_area

        # Eine Ellipse ist streng konvex, Solidity liegt also sehr nahe an 1.0
        if solidity > best_solidity:
            best_solidity = solidity
            best_coin_contour = c

    return best_coin_contour, best_solidity


def _finde_inbus(contours, cx, cy, radius):
    best, max_area = None, 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        x_c, y_c, w, h = cv2.boundingRect(c)
        dist = np.sqrt((x_c + w//2 - cx)**2 + (y_c + h//2 - cy)**2)
        if dist < radius:
            continue
        if area > max_area:
            max_area = area
            best = c
    return best


def _save_side_by_side(left, title_l, right, title_r, path, cmap_right=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(left);              axes[0].set_title(title_l); axes[0].axis("off")
    axes[1].imshow(right, cmap=cmap_right); axes[1].set_title(title_r); axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_single(img_rgb, title, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(img_rgb)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _speichere_ergebnisse(e, muenz_mm, ordner):
    pfad = os.path.join(ordner, "ergebnisse.txt")
    with open(pfad, "w", encoding="utf-8") as f:
        f.write("=" * 45 + "\n")
        f.write("   MESSERGEBNISSE - Inbusschlüssel\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Kalibrierung:\n")
        f.write(f"  Münzdurchmesser (real): {muenz_mm} mm\n")
        f.write(f"  Münzradius (Pixel):     {e['muenz_radius_px']} px\n")
        f.write(f"  Pixel pro mm:           {e['pixels_per_mm']} px/mm\n")
        # HIER WURDE CIRCULARITY DURCH SOLIDITY ERSETZT:
        f.write(f"  Solidity:               {e['solidity']}\n\n")
        f.write(f"Inbusschlüssel:\n")
        f.write(f"  Breite:  {e['breite_mm']} mm\n")
        f.write(f"  Länge:   {e['laenge_mm']} mm\n")