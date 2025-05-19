import cv2
import numpy as np
import os
import glob
import re
from pathlib import Path
import time, tracemalloc

# ---------------------------------------------------------------------------
#  Configurações gerais (ajuste conforme o layout do cartão)
# ---------------------------------------------------------------------------
N_COLS: int = 3             # número de colunas de questões
QUESTOES_POR_COL: int = 20  # 20 × 3 = 60 questões no total
MARGEM_COL_FRAC = 0.03      # 3 % de margem lateral ao dividir em colunas
HEADER_FRAC = 0.06          # região superior (até 6 %) contém cabeçalho A‑E
FOOTER_FRAC = 0.94          # região inferior (após 94 %) contém rodapé A‑E
AGRUP_THRESH = 15           # tolerância (px) para agrupar quadrados verticalmente
ALTURA_OUTLIER = 1.5        # > 1.5 × altura média ⇒ provavelmente ruído
MARGEM_LINHA_FRAC = 0.10    # corta 10 % da parte superior da linha
# ---------------------------------------------------------------------------

_K3 = np.ones((3, 3), np.uint8)  # Kernel morfológico para fechamento

# ---------------------------- 0. Funções utilitárias ----------------------- #

def reorder(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 pontos em sentido horário (TL, TR, BL, BR)."""
    pts = pts.reshape((4, 2)).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    add = pts.sum(1)
    diff = np.diff(pts, axis=1)
    ordered[0] = pts[np.argmin(add)]     # TL
    ordered[3] = pts[np.argmax(add)]     # BR
    ordered[1] = pts[np.argmin(diff)]    # TR
    ordered[2] = pts[np.argmax(diff)]    # BL
    return ordered

# -------------------------- 1. Localizar triângulos ------------------------ #

def _localizar_triangulos(img: np.ndarray, frac: float = 0.18):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lim_x, lim_y = int(w * frac), int(h * frac)
    marcadores: dict[str, tuple[int, int, float]] = {}

    for c in cnts:
        if len(cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)) != 3:
            continue
        M = cv2.moments(c)
        if not M["m00"]:
            continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        area = cv2.contourArea(c)

        if cx < lim_x and cy < lim_y:
            k = "TL"
        elif cx > w - lim_x and cy < lim_y:
            k = "TR"
        elif cx < lim_x and cy > h - lim_y:
            k = "BL"
        elif cx > w - lim_x and cy > h - lim_y:
            k = "BR"
        else:
            continue

        if k not in marcadores or area > marcadores[k][2]:
            marcadores[k] = (cx, cy, area)

    # devolve apenas (x, y)
    return {k: (v[0], v[1]) for k, v in marcadores.items()}

# ------------------------------ 2. Retificação ---------------------------- #

def detectar_e_retificar(img: np.ndarray):
    verts = _localizar_triangulos(img)
    if len(verts) != 4:
        print("⚠  Marcadores insuficientes – imagem ignorada (não retificada)")
        return None

    pts = reorder(np.array([verts[k] for k in ("TL", "TR", "BL", "BR")], dtype=np.float32))

    width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    height = int(max(np.linalg.norm(pts[0] - pts[2]), np.linalg.norm(pts[1] - pts[3])))

    dst = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (width, height))

# ---------------------------- 3. Dividir colunas -------------------------- #

def split_columns(img: np.ndarray, n_cols: int = N_COLS, marg_frac: float = MARGEM_COL_FRAC):
    h, w = img.shape[:2]
    m = int(w * marg_frac)
    cw = (w - 2 * m) // n_cols
    return [img[:, m + i * cw : m + (i + 1) * cw] if i < n_cols - 1 else img[:, m + i * cw : w - m] for i in range(n_cols)]

# --------------------------- 4. Quadrados (bolhas) ------------------------- #

def _square_contours(col_img: np.ndarray):
    gray = cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, _K3, iterations=1)

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sq = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 10 < w < 60 and 0.75 <= w / h <= 1.3:
            sq.append((y + h // 2, x, y, h))  # (centroY, x, y, h)
    # ordena topo → base
    sq.sort(key=lambda s: s[0])
    # remove quadrados anormalmente altos
    if sq:
        avg_h = np.mean([h for _, _, _, h in sq])
        sq = [s for s in sq if s[3] < ALTURA_OUTLIER * avg_h]
    return sq

# ------------------------- 5. Dividir em linhas --------------------------- #

def split_rows(col_img: np.ndarray, expected: int = QUESTOES_POR_COL):
    """Retorna lista (pode estar incompleta) de linhas detectadas."""
    sq = _square_contours(col_img)
    if not sq:
        return []

    # 5.1 Agrupamento vertical
    clusters: list[list[tuple[int, int, int]]] = []
    for cy, _, y, h in sq:
        if not clusters or cy - clusters[-1][-1][0] > AGRUP_THRESH:
            clusters.append([(cy, y, h)])
        else:
            clusters[-1].append((cy, y, h))

    clusters = [cl for cl in clusters if len(cl) >= 5]
    if not clusters:
        return []

    # 5.2 Remove cabeçalho/rodapé
    h_col = col_img.shape[0]
    clusters = [cl for cl in clusters if HEADER_FRAC * h_col < cl[0][0] < FOOTER_FRAC * h_col]
    if not clusters:
        return []

    # 5.3 Ordena e, se houver sobra, descarta extremos
    clusters.sort(key=lambda c: c[0][0])
    while len(clusters) > expected:
        clusters.pop(0)
        if len(clusters) > expected:
            clusters.pop()

    # 5.4 Recorte
    linhas = []
    for cl in clusters:
        top = min(y for _, y, _ in cl)
        bottom = max(y + h for _, y, h in cl)
        margin = int((bottom - top) * MARGEM_LINHA_FRAC)
        linhas.append(col_img[top + margin : bottom, :])

    return linhas

# ------------------------------- 6. Pipeline ------------------------------ #

def processar_cartoes(pasta_glob: str = r"cartoes\\img_anonimizado\\*.jpg"):
    total_esperado = QUESTOES_POR_COL * N_COLS

    for image_path in glob.glob(pasta_glob):
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Não foi possível abrir {image_path}")
            continue

        # 1. Retificação
        warp = detectar_e_retificar(img)
        if warp is None:
            continue  # já avisado

        # 2. Colunas
        cols = split_columns(warp)
        candidato_id = re.sub(r"\D", "", os.path.basename(image_path)) or "cartao"
        out_dir = Path("processamento_de_cartoes_de_imagem") / "questoes" / candidato_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # 3. Linhas – salva mesmo quando incompleto
        q = 1
        for idx, col in enumerate(cols, start=1):
            rows = split_rows(col)
            if not rows:
                print(f"⚠ Coluna {idx}: nenhuma linha detectada em {image_path} – nada salvo")
                continue
            if len(rows) != QUESTOES_POR_COL:
                print(f"⚠ Coluna {idx}: apenas {len(rows)} de {QUESTOES_POR_COL} linhas detectadas em {image_path} – salvando assim mesmo")
            for r in rows:
                cv2.imwrite(str(out_dir / f"Q{q:02d}.png"), r)
                q += 1

        # 4. Feedback geral do cartão
        salvos = q - 1
        if salvos == total_esperado:
            print(f"✅ {candidato_id}: {salvos} questões salvas corretamente em {out_dir}")
        elif salvos > 0:
            print(f"⚠ {candidato_id}: somente {salvos}/{total_esperado} questões salvas – arquivos parciais disponíveis em {out_dir}")
        else:
            print(f"❌ {candidato_id}: nenhuma questão pôde ser salva")

        # 5. Visualização rápida (opcional)
        cv2.imshow("Cartão Retificado", warp)
        cv2.waitKey(300)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    processar_cartoes()
