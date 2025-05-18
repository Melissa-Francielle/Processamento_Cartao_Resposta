import cv2
import numpy as np
import os
import glob
import re
from pathlib import Path

def reorder(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape((4, 2)).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    add = pts.sum(1)
    diff = np.diff(pts, axis=1)
    ordered[0] = pts[np.argmin(add)]     # TL
    ordered[3] = pts[np.argmax(add)]     # BR
    ordered[1] = pts[np.argmin(diff)]    # TR
    ordered[2] = pts[np.argmax(diff)]    # BL
    return ordered

def _localizar_triangulos(img: np.ndarray, frac: float = 0.18) -> dict[str, tuple[int,int]]:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)

    lim_x, lim_y = int(w*frac), int(h*frac)
    m: dict[str, tuple[int,int,float]] = {}

    for c in cnts:
        if len(cv2.approxPolyDP(c, 0.04*cv2.arcLength(c, True), True)) != 3:
            continue
        M = cv2.moments(c)
        if not M["m00"]:
            continue
        cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        area = cv2.contourArea(c)

        if cx < lim_x and cy < lim_y:            k = "TL"
        elif cx > w - lim_x and cy < lim_y:      k = "TR"
        elif cx < lim_x and cy > h - lim_y:      k = "BL"
        elif cx > w - lim_x and cy > h - lim_y:  k = "BR"
        else:                                    continue

        if k not in m or area > m[k][2]:
            m[k] = (cx, cy, area)

    return {k: (v[0], v[1]) for k, v in m.items()}

def detectar_e_retificar(img: np.ndarray) -> np.ndarray|None:
    verts = _localizar_triangulos(img)
    if len(verts) != 4:
        print(f"⚠ Marcadores insuficientes. Encontrados: {len(verts)}/4")
        return None

    pts = reorder(np.array([verts[k] for k in ("TL","TR","BL","BR")],
                          dtype=np.float32))

    width = int(max(np.linalg.norm(pts[0]-pts[1]),
                   np.linalg.norm(pts[2]-pts[3])))
    height = int(max(np.linalg.norm(pts[0]-pts[2]),
                    np.linalg.norm(pts[1]-pts[3])))

    dst = np.float32([[0,0],[width-1,0],[0,height-1],[width-1,height-1]])
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (width, height))

def split_columns(img: np.ndarray, n_cols:int=3, marg_frac:float=0.03):
    h, w = img.shape[:2]
    m = int(w*marg_frac)
    cw = (w - 2*m) // n_cols
    return [img[:, m+i*cw : m+(i+1)*cw] if i < n_cols-1
           else img[:, m+i*cw : w-m] for i in range(n_cols)]

def _square_contours(col_img):
    gray = cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)
    thr  = cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr  = cv2.morphologyEx(thr, cv2.MORPH_CLOSE,
                            np.ones((3,3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    sq = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w > 10 and 0.75 <= w/h <= 1.3:
            cy = y + h // 2
            sq.append((cy, x, y, h))
    return sorted(sq, key=lambda s: s[0])

def split_rows(col_img: np.ndarray, expected: int = 20):
    sq = _square_contours(col_img)
    if not sq:
        return None

    # Filtro adicional: remover quadrados muito altos (possivelmente cabeçalho)
    avg_height = np.mean([h for _, _, _, h in sq])
    sq = [s for s in sq if s[3] < 1.5 * avg_height]

    # Agrupamento vertical
    clusters = []
    thresh = 15
    for cy, x, y, h in sq:
        if not clusters or cy - clusters[-1][-1][0] > thresh:
            clusters.append([(cy, y, h)])
        else:
            clusters[-1].append((cy, y, h))

    # Filtro: apenas clusters com pelo menos 5 quadrados (A-E)
    clusters = [c for c in clusters if len(c) >= 5]
    
    # Ordenar pela posição vertical
    clusters.sort(key=lambda c: c[0][0])
    
    # Remover o primeiro cluster se ele for o cabeçalho (A B C D E)
    if len(clusters) > expected:
        clusters = clusters[1:]

    # Pegar apenas os primeiros 'expected' clusters
    clusters = clusters[:expected]

    # Calcula áreas das linhas
    linhas = []
    for cl in clusters:
        top = min(y for _, y, _ in cl)
        bottom = max(y + h for _, y, h in cl)
        h_linha = bottom - top
        linhas.append((top, bottom, h_linha))

    if len(linhas) < expected:
        return None

    # Descarta extremos com base na mediana
    mediana = np.median([h for _, _, h in linhas])
    final = []
    for (top, bottom, h_linha) in linhas:
        if 0.7 * mediana <= h_linha <= 1.3 * mediana:
            # Cortar um pouco da parte superior para evitar cabeçalho
            margin = int(h_linha * 0.1)
            final.append(col_img[top+margin:bottom, :])

    return final[:expected]
# -------------------------------------------------
for image_path in glob.glob(r"cartoes\img_anonimizado\*.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Falha ao abrir {image_path}")
        continue

    # 1. Retificação
    warp = detectar_e_retificar(img)
    if warp is None:
        cv2.imshow("Imagem com Problema", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        continue

    # 2. Divisão em colunas
    cols = split_columns(warp)
    
    candidato_id = re.sub(r'\D', '', os.path.basename(image_path)) or "cartao"
    out_dir = Path("processamento_de_cartoes_de_imagem") / "questoes" / candidato_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Processar cada coluna e salvar questões
    q = 1
    for col in cols:
        rows = split_rows(col)
        if rows is None or len(rows) != 20:
            print(f"⚠ Falha segmentar linhas em {image_path} - encontradas {len(rows) if rows else 0} linhas")
            break
        for r in rows:
            cv2.imwrite(str(out_dir / f"Q{q:02d}.png"), r)
            q += 1

    if q == 61:  # Verifica se todas as 60 questões foram processadas (q termina em 61)
        print(f"✅ {candidato_id}: Todas as 60 questões salvas em {out_dir}")
    elif q > 1:
        print(f"⚠ {candidato_id}: Apenas {q-1} questões salvas (esperado 60)")

    cv2.imshow("Cartão Retificado", warp)
    cv2.waitKey(500)
    cv2.destroyAllWindows()