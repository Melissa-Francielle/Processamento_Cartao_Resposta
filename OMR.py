import cv2
import numpy as np

def reorder(points):
    # Função para reordenar os pontos na ordem: superior esquerdo, superior direito, inferior esquerdo, inferior direito
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points

image_path = r"cartoes\img_anonimizado\01000101.jpg"
width = 800
height = 800

img = cv2.imread(image_path)
img = cv2.resize(img, (width, height))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

triangulos = []
h, w = img.shape[:2]
margem = 300

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 50:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) == 3:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                if (cx < margem and cy < margem):
                    triangulos.append(("superior esquerdo", (cx, cy)))
                elif (cx > w - margem and cy < margem):
                    triangulos.append(("superior direito", (cx, cy)))
                elif (cx < margem and cy > h - margem):
                    triangulos.append(("inferior esquerdo", (cx, cy)))
                elif (cx > w - margem and cy > h - margem):
                    triangulos.append(("inferior direito", (cx, cy)))
                cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)

# Se encontrou os 4 marcadores, aplicar a transformação de perspectiva
if len(triangulos) == 4:
    # Ordenar os triângulos na ordem correta
    triangulos.sort(key=lambda x: x[0])
    
    # Extrair os pontos centrais dos triângulos
    gradePoints = np.array([t[1] for t in triangulos], dtype=np.float32).reshape(-1, 1, 2)
    
    # Definir os pontos do maior contorno (ROI do cartão - assumindo que ocupa quase toda a imagem)
    biggestContour = np.array([
        [margem, margem],
        [w - margem, margem],
        [margem, h - margem],
        [w - margem, h - margem]
    ], dtype=np.float32).reshape(-1, 1, 2)
    
    
    # Reordenar os pontos
    biggestContour = reorder(biggestContour)
    gradePoints = reorder(gradePoints)
    
    pt1 = np.float32(gradePoints)
    pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (width, height))
    
    # Mostrar resultados
    #cv2.imshow("Marcadores e ROI", img)
    cv2.imshow("Cartão Retificado", imgWarpColored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"Marcadores insuficientes. Encontrados: {len(triangulos)}/4")
    cv2.imshow("Triângulos detectados", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
