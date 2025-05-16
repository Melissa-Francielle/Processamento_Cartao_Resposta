import cv2
import numpy as np
import os
import glob
import re  

def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points

def split_columns(img, n_cols=3, margin=5):
    h, w = img.shape[:2]
    col_w = w // n_cols
    cols = []
    for i in range(n_cols):
        x0 = i * col_w + margin
        x1 = (i + 1) * col_w - margin if i < n_cols - 1 else w - margin
        cols.append(img[:, x0:x1])
    return cols

def split_rows(col_img, n_rows=20, margin=5):
    h, w = col_img.shape[:2]
    row_h = h // n_rows
    rows = []
    for i in range(n_rows):
        y0 = i * row_h + margin
        y1 = (i + 1) * row_h - margin if i < n_rows - 1 else h - margin
        rows.append(col_img[y0:y1, :])
    return rows

def detectar_e_retificar(img, largura, altura, margem):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    triangulos = []
    h, w = img.shape[:2]

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

    if len(triangulos) == 4:
        triangulos.sort(key=lambda x: x[0])
        gradePoints = np.array([t[1] for t in triangulos], dtype=np.float32).reshape(-1, 1, 2)

        # Pontos alvo para mapeamento
        pt1 = reorder(gradePoints)
        pt2 = np.float32([[0, 0], [largura, 0], [0, altura], [largura, altura]])

        matrix = cv2.getPerspectiveTransform(np.float32(pt1), pt2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (largura, altura))
        return imgWarpColored
    else:
        print(f"Marcadores insuficientes. Encontrados: {len(triangulos)}/4")
        return None

# -------------------------------------------------
width = 800
height = 800
margem = 300

for image_path in glob.glob(r"cartoes\img_anonimizado\*.jpg"):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (width, height))

    imgWarpColored = detectar_e_retificar(img, width, height, margem)

    if imgWarpColored is None:
        cv2.imshow("Imagem com Problema", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        continue

    colunas = split_columns(imgWarpColored)

    candidato_id = re.sub(r'\D', '', os.path.basename(image_path))
    out_dir = os.path.join("processamento_de_cartoes_de_imagem", "colunas", candidato_id)
    os.makedirs(out_dir, exist_ok=True)

    # Mostrar cartão retificado inteiro
    cv2.imshow("Cartão Retificado", imgWarpColored)

    # Processar cada coluna
    for idx, col in enumerate(colunas, start=1):
        # Salvar coluna
        cv2.imwrite(os.path.join(out_dir, f"coluna{idx}.png"), col)
        
        # Dividir coluna em linhas (questões)
        linhas = split_rows(col)
        
        # Criar subpasta para as questões desta coluna
        questao_dir = os.path.join(out_dir, f"coluna{idx}_questoes")
        os.makedirs(questao_dir, exist_ok=True)
        
        # Salvar cada questão
        for q_idx, questao in enumerate(linhas, start=1):
            questao_num = q_idx + (idx-1)*20  # Calcula o número da questão (1-60)
            cv2.imwrite(os.path.join(questao_dir, f"questao_{questao_num:02d}.png"), questao)
            
            # Mostrar questão (opcional)
            cv2.imshow(f"Coluna {idx} - Questão {questao_num}", questao)
            cv2.waitKey(50)  # Pequeno delay para visualização

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"{candidato_id}: colunas e questões salvas em {out_dir}")
