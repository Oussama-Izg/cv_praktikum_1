import cv2
import numpy as np
from helper import find_inbus_contour

img = cv2.imread("bilder/image_1.jpg")
# Bild in Graustufen umwandeln
imgInGrey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Bild weichzeichnen um Rauschen zu reduzieren
imgBlurred = cv2.GaussianBlur(imgInGrey, (7, 7), 0)

# Binarisieren (Schwellwertbildung)
#_, imgBinary = cv2.threshold(imgBlurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#imgBinary = cv2.adaptiveThreshold(imgBlurred,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
#imgBinary = cv2.bitwise_not(imgBinary)
_, imgBinary = cv2.threshold(imgBlurred, 110, 255, cv2.THRESH_BINARY)

#cv2.imshow("Binary Image", imgBinary)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


kernel = np.ones((7, 7), np.uint8)

#Dilatation → Erosion: schließt Löcher im Objekt
imgClean = cv2.morphologyEx(imgBinary, cv2.MORPH_CLOSE, kernel, iterations=3)

#Erosion → Dilatation: entfernt Rauschen (weiße Punkte)
imgClean = cv2.morphologyEx(imgClean, cv2.MORPH_OPEN, kernel, iterations=3)

edges = cv2.Canny(imgBlurred, 50, 150)

#cv2.imshow("Clean Image", imgClean)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Konturen finden
contours, _ = cv2.findContours(imgClean, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

inbus_contour = find_inbus_contour(contours)

if inbus_contour is None:
    exit()

# Konturpunkte extrahieren
data_pts = np.squeeze(inbus_contour).astype(np.float32)

# PCA berechnen
mean, eigenvectors, eigenvalues = cv2.PCACompute2(
    data_pts,
    mean=np.array([])
)

# Hauptachse bestimmen
center = mean[0]
main_axis = eigenvectors[0]
secondary_axis = eigenvectors[1]

angle = np.arctan2(main_axis[1], main_axis[0])
angle_deg = np.degrees(angle)

# Debug-Bild
debug = img.copy()

# Mittelpunkt zeichnen
cx, cy = int(center[0]), int(center[1])
cv2.circle(debug, (cx, cy), 8, (0, 0, 255), -1)

# Länge der Achsen für Darstellung
scale = 150

# Hauptachse
x1 = int(cx + main_axis[0] * scale)
y1 = int(cy + main_axis[1] * scale)

# Nebenachse
x2 = int(cx + secondary_axis[0] * scale)
y2 = int(cy + secondary_axis[1] * scale)

# Hauptachse zeichnen (grün)
cv2.line(debug, (cx, cy), (x1, y1), (0, 255, 0), 3)

# Nebenachse zeichnen (blau)
cv2.line(debug, (cx, cy), (x2, y2), (255, 0, 0), 3)

# Kontur zeichnen
cv2.drawContours(debug, [inbus_contour], -1, (0, 255, 255), 2)

print(f"Winkel: {angle_deg:.2f} Grad")

cv2.imshow("PCA Analyse", debug)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
# Debug
debug = img.copy()
cv2.drawContours(debug, contours, -1, (0, 255, 0), 1)   # alle grün
cv2.drawContours(debug, [inbus_contour], -1, (0, 0, 255), 3)    # Inbus rot
cv2.imshow("Erkannter Inbus", debug)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


