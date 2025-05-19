#!/usr/bin/env python
# ────────────────────────────────────────────────────────────────────────────
# verificar_respostas.py  –  v2
#
# Uso rápido:
#   python verificar_respostas.py                           # só gera respostas_candidatos.csv
#   python verificar_respostas.py -g respostas_.csv           # gera respostas + compara e cria comparacao_resumo.csv
#   python verificar_respostas.py -a marcado.csv -g respostas_.csv   # usa CSV já existente nas pastas
# Opções:
#   -a/--arquivo   Nome de um CSV de respostas já salvo em cada pasta (ex.: marcado.csv)
#   -g/--respostas_  Caminho para o respostas_ oficial (se omitido, não há comparação)
#   -o/--out       Nome do CSV agregado de respostas (padrão: respostas_candidatos.csv)
# ────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import argparse, re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ──────────────────────────── Configurações globais ─────────────────────────
SEL_DIR      = Path("processamento_de_cartoes_de_imagem/questoes")          # raiz das pastas de candidatos
ARQ_PADRAO   = "respostas_.csv"               # csv de respostas dentro da pasta
LETRAS       = list("ABCDE")                 # alternativas possíveis
LIMIAR_FILL  = 0.40                          # % mínimo de preenchimento (miolo do quadrado)

# ───────────────────────────── Funções de OMR ──────────────────────────────
def extrair_resposta_imagem(img_path: Path) -> str | None:
    """
    Lê Qxx.png e devolve a letra marcada ('A'-'E') ou None se nada marcado.
    Estratégia: achar contornos ~quadrados e medir o preenchimento no miolo.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠  Falha ao abrir {img_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr  = cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []          # (x, y, w, h, fill_ratio)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if not (15 < w < 70 and 0.75 < w/h < 1.25):
            continue     # descarta se não for quadradinho provável

        # avalia preenchimento só no centro (20-80 %)
        dx, dy = int(w*0.2), int(h*0.2)
        core = thr[y+dy:y+h-dy, x+dx:x+w-dx]
        if core.size == 0:
            continue
        fill = cv2.countNonZero(core) / core.size
        quads.append((x, y, w, h, fill))

    if len(quads) < 5:
        # fallback simples – divide a largura da imagem em 5 colunas iguais
        h_, w_ = thr.shape
        step = w_ // 5
        for i in range(5):
            x0 = i*step
            core = thr[:, x0+int(step*0.2):x0+int(step*0.8)]
            fill = cv2.countNonZero(core) / core.size
            quads.append((x0, 0, step, h_, fill))

    # mantém os 5 contornos mais à direita e ordena da esquerda p/ direita
    quads.sort(key=lambda q: q[0])
    quads = quads[-5:]
    quads.sort(key=lambda q: q[0])

    fills = [q[4] for q in quads]
    idx_validos = [i for i, f in enumerate(fills) if f >= LIMIAR_FILL]
    if not idx_validos:
        return None
    idx_escolhido = max(idx_validos, key=lambda i: fills[i])
    return LETRAS[idx_escolhido]

# ───────────────────── Funções de carga/geração de respostas ───────────────
def carregar_respostas_csv(pasta: Path, nome: str) -> pd.DataFrame | None:
    arq = pasta / nome
    if not arq.exists():
        return None
    try:
        df = pd.read_csv(arq, dtype={"questao": int, "resposta": str})
        df["resposta"] = df["resposta"].str.strip().str.upper()
        return df
    except Exception as e:
        print(f"⚠  Erro lendo {arq}: {e}")
        return None

def gerar_respostas_de_imagens(pasta: Path) -> pd.DataFrame:
    linhas = []
    for img_path in sorted(pasta.glob("Q*.png")):
        m = re.match(r"Q(\d{1,3})", img_path.stem, re.I)
        if not m:
            continue
        qnum   = int(m.group(1))
        letra  = extrair_resposta_imagem(img_path) or "-"
        linhas.append({"questao": qnum, "resposta": letra})
    return pd.DataFrame(linhas)

def obter_respostas(pasta: Path, csv_name: str | None) -> pd.DataFrame | None:
    """Tenta (1) arquivo fornecido, (2) respostas.csv, (3) OMR das imagens."""
    if csv_name:
        if (df := carregar_respostas_csv(pasta, csv_name)) is not None:
            return df
    if (df := carregar_respostas_csv(pasta, ARQ_PADRAO)) is not None:
        return df
    return gerar_respostas_de_imagens(pasta)

# ───────────────────────── Funções de comparação (opcional) ────────────────
def carregar_respostas_(path: Path) -> pd.DataFrame:
    gb = pd.read_csv(path, dtype={"questao": int, "resposta": str, "candidato": int})
    gb["resposta"] = gb["resposta"].str.strip().str.upper()
    return gb

def comparar(respostas_: pd.DataFrame, respostas: pd.DataFrame, cand_id: int) -> pd.DataFrame:
    gb_cand = respostas_[respostas_["candidato"] == cand_id]
    if gb_cand.empty:
        raise ValueError(f"Candidato {cand_id} não consta no respostas_.")
    merged = gb_cand.merge(respostas, on="questao", how="left", suffixes=("_ofic", "_marc"))
    merged["correta"] = merged["resposta_ofic"] == merged["resposta_marc"]
    return merged

# ───────────────────────────── Utils & Main ────────────────────────────────
def extrair_id_pasta(nome: str) -> int | None:
    """Extrai os três dígitos centrais – padrão 010xxx01 → xxx."""
    m = re.match(r"010(\d{3})01$", nome)
    return int(m.group(1)) if m else None

def main() -> None:
    ap = argparse.ArgumentParser(description="Extrai respostas e compara com respostas_ (opcional).")
    ap.add_argument("-a", "--arquivo", help="Nome do CSV de respostas dentro das pastas (ex.: marcado.csv).")
    ap.add_argument("-g", "--respostas_", help="CSV de respostas_ oficial para comparação.")
    ap.add_argument("-o", "--out", default="respostas_candidatos.csv",
                    help="Nome do CSV agregado (default: respostas_candidatos.csv).")
    args = ap.parse_args()

    if not SEL_DIR.exists():
        print(f"❌  Pasta '{SEL_DIR}' não encontrada.")
        return

    # Se for comparar, carrega o respostas_
    gb = None
    if args.respostas_:
        g_path = Path(args.respostas_)
        if not g_path.exists():
            print(f"❌  respostas_ '{g_path}' não encontrado.")
            return
        gb = carregar_respostas_(g_path)

    respostas_gerais = []   # lista de dicionários para o CSV final
    resumo_cmp        = []   # resumo de acertos se houver respostas_

    for pasta in sorted(SEL_DIR.iterdir()):
        if not pasta.is_dir():
            continue
        cand_id = extrair_id_pasta(pasta.name)
        if cand_id is None:
            print(f"⚠  Pasta ignorada (nome fora do padrão): {pasta.name}")
            continue

        resp_df = obter_respostas(pasta, args.arquivo)
        if resp_df is None or resp_df.empty:
            print(f"⚠  Nenhuma resposta extraída para {pasta.name}.")
            continue

        # adiciona ao CSV agregado
        for _, row in resp_df.iterrows():
            respostas_gerais.append({"candidato": cand_id,
                                      "questao"  : int(row["questao"]),
                                      "resposta" : row["resposta"]})

        # se houver respostas_, faz a comparação
        if gb is not None:
            try:
                cmp_df = comparar(gb, resp_df, cand_id)
            except ValueError as e:
                print(e)
                continue

            acertos = int(cmp_df["correta"].sum())
            total   = len(cmp_df)
            perc    = acertos / total * 100 if total else 0
            resumo_cmp.append({"candidato": cand_id, "acertos": acertos,
                               "total": total, "%": f"{perc:.1f}"})

            erros = cmp_df[~cmp_df["correta"]]
            if not erros.empty:
                print(f"\n❌ Divergências – cand {cand_id} (pasta {pasta.name}):")
                print(erros[["questao","resposta_ofic","resposta_marc"]].to_string(index=False))

    # ── salva CSV de respostas ────────────────────────────────────────────
    if not respostas_gerais:
        print("Nenhum candidato processado – nada a salvar.")
        return
    resp_df_final = pd.DataFrame(respostas_gerais).sort_values(["candidato","questao"])
    resp_df_final.to_csv(args.out, index=False)
    print(f"\n✔  '{args.out}' salvo com {len(resp_df_final)} linhas.")

    # ── se for o caso, salva também o resumo de comparação ───────────────
    if gb is not None:
        if not resumo_cmp:
            print("\n⚠  Nenhum candidato pôde ser comparado ao respostas_.")
            return
        resumo_df = pd.DataFrame(resumo_cmp).sort_values("candidato")
        resumo_df.to_csv("comparacao_resumo.csv", index=False)
        print("\nResumo de acertos por candidato:")
        print(resumo_df.to_string(index=False))
        print("\n✔  'comparacao_resumo.csv' salvo.")

if __name__ == "__main__":
    main()
