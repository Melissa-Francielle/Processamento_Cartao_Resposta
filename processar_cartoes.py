import cv2
import numpy as np
import os, glob, re
from pathlib import Path

# ╭────────────────────────────────────────────────────────────╮
# │            Funções utilitárias de geometria               │
# ╰────────────────────────────────────────────────────────────╯
def reorder(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 pontos (TL, TR, BL, BR) e devolve shape (4,2)."""
    pts = pts.reshape((4, 2)).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    add  = pts.sum(1)
    diff = np.diff(pts, axis=1)
    ordered[0] = pts[np.argmin(add)]     # TL
    ordered[3] = pts[np.argmax(add)]     # BR
    ordered[1] = pts[np.argmin(diff)]    # TR
    ordered[2] = pts[np.argmax(diff)]    # BL
    return ordered

# ╭────────────────────────────────────────────────────────────╮
# │      1) localizar triângulos em cada canto da imagem      │
# ╰────────────────────────────────────────────────────────────╯
def _localizar_triangulos(img: np.ndarray,
                          frac: float = 0.18) -> dict[str, tuple[int,int]]:
    h, w = img.shape[:2]
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        area   = cv2.contourArea(c)

        if   cx < lim_x and cy < lim_y:            k = "TL"
        elif cx > w - lim_x and cy < lim_y:        k = "TR"
        elif cx < lim_x and cy > h - lim_y:        k = "BL"
        elif cx > w - lim_x and cy > h - lim_y:    k = "BR"
        else:                                      continue

        if k not in m or area > m[k][2]:
            m[k] = (cx, cy, area)

    return {k: (v[0], v[1]) for k, v in m.items()}

# ╭────────────────────────────────────────────────────────────╮
# │                 2) retificação (homografia)               │
# ╰────────────────────────────────────────────────────────────╯
def detectar_e_retificar(img: np.ndarray, frac: float = 0.18) -> np.ndarray|None:
    verts = _localizar_triangulos(img, frac)
    if len(verts) != 4:
        print(f"⚠  Marcadores insuficientes. Encontrados: {len(verts)}/4")
        return None

    pts = reorder(np.array([verts[k] for k in ("TL","TR","BL","BR")],
                            dtype=np.float32))

    width  = int(max(np.linalg.norm(pts[0]-pts[1]),
                     np.linalg.norm(pts[2]-pts[3])))
    height = int(max(np.linalg.norm(pts[0]-pts[2]),
                     np.linalg.norm(pts[1]-pts[3])))

    dst = np.float32([[0,0],[width-1,0],[0,height-1],[width-1,height-1]])
    M   = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (width, height))

# ╭────────────────────────────────────────────────────────────╮
# │   3)  divide em 3 colunas preservando margem interna       │
# ╰────────────────────────────────────────────────────────────╯
def split_columns(img: np.ndarray, n_cols:int=3, marg_frac:float=0.03):
    h, w = img.shape[:2]
    m  = int(w*marg_frac)
    cw = (w - 2*m) // n_cols
    return [img[:, m+i*cw : m+(i+1)*cw] if i < n_cols-1
            else img[:, m+i*cw : w-m]  for i in range(n_cols)]

# ╭────────────────────────────────────────────────────────────╮
# │  4) detecção de linhas via ≥5 quadrados alinhados          │
# ╰────────────────────────────────────────────────────────────╯
def _square_contours(col_img: np.ndarray):
    gray = cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)
    thr  = cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    sq = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if 10 < w < 60 and 0.7 <= w/h <= 1.3:
            sq.append((y+h//2, x, y, h))  # (centroY, x, y, h)
    return sorted(sq, key=lambda s: s[0])

def split_rows(col_img: np.ndarray, expected: int = 20):
    """Recorta exatamente 20 linhas (número + 5 bolhas)."""
    sq = _square_contours(col_img)
    if not sq:
        return None

    # ── 1. Agrupamento vertical ─────────────────────────────────────────
    clusters = []
    thresh = 15                          # tolerância em pixels (eixo Y)
    for cy, x, y, h in sq:
        if not clusters or cy - clusters[-1][-1][0] > thresh:
            clusters.append([(cy, y, h)])
        else:
            clusters[-1].append((cy, y, h))

    # ── 2. Constrói bounding-box dos clusters com ≥5 quadrados ──────────
    lines = []
    for cl in clusters:
        if len(cl) >= 5:                 # considera linha “real”
            top    = min(y for _, y, _ in cl)
            bottom = max(y + h for _, y, h in cl)
            lines.append(col_img[top:bottom, :])

    if not lines:
        return None

    # ── 3. Remove cabeçalho / rodapé se houver sobras ───────────────────
    while len(lines) > expected:
        lines.pop(0)                     # remove o primeiro  (cabeçalho)
        if len(lines) > expected:
            lines.pop()                 # remove o último   (rodapé)

    # Se ainda não conseguirmos 20, aborta para fallback
    if len(lines) != expected:
        return None

    return lines

# ╭────────────────────────────────────────────────────────────╮
# │                   5) programa principal                    │
# ╰────────────────────────────────────────────────────────────╯
def main():
    IN_GLOB = r"cartoes/img_anonimizado/*.jpg"
    OUT_ROOT = Path("processamento_de_cartoes_de_imagem/questoes")

    for path in glob.glob(IN_GLOB):
        img = cv2.imread(path)
        if img is None:
            print(f"❌  Falha ao abrir {path}")
            continue

        warp = detectar_e_retificar(img)
        if warp is None:
            continue

        cols = split_columns(warp)
        cand = re.sub(r"\D", "", os.path.basename(path)) or "cartao"
        out_dir = OUT_ROOT / cand
        out_dir.mkdir(parents=True, exist_ok=True)

        q = 1
        for col in cols:
            rows = split_rows(col)
            if rows is None:
                print(f"⚠  Falha segmentar linhas em {path}")
                break
            for r in rows:
                cv2.imwrite(str(out_dir / f"Q{q:02d}.png"), r)
                q += 1

        if q > 1:
            print(f"✅ {cand}: {q-1} questões salvas em {out_dir}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()