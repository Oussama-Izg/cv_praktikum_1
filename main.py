import cv2
import math
import matplotlib.pyplot as plt


def zeige_bild(bilder, titel=None, texte=None, spalten=None, figsize=None, cmap="gray"):
    if not isinstance(bilder, (list, tuple)):
        bilder = [bilder]

    anzahl = len(bilder)

    if titel is None:
        titel = ["Bild"] * anzahl
    elif isinstance(titel, str):
        titel = [titel]
    else:
        titel = list(titel)

    if texte is None:
        texte = [""] * anzahl
    elif isinstance(texte, str):
        texte = [texte]
    else:
        texte = list(texte)

    titel += [""] * (anzahl - len(titel))
    texte += [""] * (anzahl - len(texte))

    if spalten is None:
        spalten = min(anzahl, 3)

    zeilen = math.ceil(anzahl / spalten)

    if figsize is None:
        figsize = (5 * spalten, 4 * zeilen)

    fig, axes = plt.subplots(zeilen, spalten, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for index, bild in enumerate(bilder):
        ax = axes[index]
        if bild.ndim == 2:
            ax.imshow(bild, cmap=cmap)
        else:
            ax.imshow(bild)
        ax.set_title(titel[index])
        if texte[index]:
            ax.text(0.5, -0.08, texte[index], ha="center", va="top", transform=ax.transAxes)
        ax.axis("off")

    for ax in axes[anzahl:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    # Bild laden
    image = cv2.imread("bilder/1.jpg")

    # Prüfen, ob das Bild geladen wurde
    if image is None:
        print("Fehler: Bild konnte nicht geladen werden.")
        return

    # In Graustufen umwandeln
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Glätten, um Rauschen vor der Kantenerkennung zu reduzieren
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarisieren
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Schritt 2: unterschiedliche Kantenerkennungsmethoden auf dem binarisierten Bild vergleichen
    canny_edges = cv2.Canny(binary, threshold1=75, threshold2=200)

    sobel_x = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.convertScaleAbs(cv2.addWeighted(
        cv2.convertScaleAbs(sobel_x), 0.5,
        cv2.convertScaleAbs(sobel_y), 0.5,
        0,
    ))

    laplacian_edges = cv2.convertScaleAbs(cv2.Laplacian(binary, cv2.CV_64F))

    scharr_x = cv2.Scharr(binary, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(binary, cv2.CV_64F, 0, 1)
    scharr_edges = cv2.convertScaleAbs(cv2.addWeighted(
        cv2.convertScaleAbs(scharr_x), 0.5,
        cv2.convertScaleAbs(scharr_y), 0.5,
        0,
    ))

    zeige_bild(
        [gray, blurred, binary],
        titel=["Graustufen", "Geglaettet", "Binarisiert"],
        figsize=(12, 5),
    )

    zeige_bild(
        [canny_edges, sobel_edges, laplacian_edges, scharr_edges],
        titel=["Canny auf binary", "Sobel auf binary", "Laplacian auf binary", "Scharr auf binary"],
        spalten=2,
        figsize=(12, 8),
    )


if __name__ == "__main__":
    main()
