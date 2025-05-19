# comparar_respostas.py
# =========================================================
#  Compara respostas de dois CSVs:
#    • verifica correspondência 1 ↔ 01000101, 2 ↔ 01000201 …
#    • calcula acurácia, tempo e pico de memória
# ---------------------------------------------------------
import argparse
import csv
import re
import sys
import time
import tracemalloc
from pathlib import Path

import pandas as pd


# ╭───────────────────────────────╮
# │ 1. Utilidades de candidato    │
# ╰───────────────────────────────╯
PADRAO_CODE = re.compile(r"^010\d{3}01$")


def numeric2code(n: int) -> str:
    """1 → '01000101', 253 → '01025301'."""
    return f"010{n:03d}01"


def get_numeric_id(val: str) -> int:
    """
    Converte qualquer representação (numérica ou código) em ID inteiro.

    - '1' → 1
    - '01000101' → 1
    """
    s = str(val).strip()
    if s.isdigit():
        if len(s) == 8:            # código completo
            return int(s[3:6])     # dígitos centrais
        return int(s)              # número simples
    raise ValueError(f"Formato de candidato desconhecido: {val}")


# ╭───────────────────────────────╮
# │ 2. Carregar CSV (sep auto)    │
# ╰───────────────────────────────╯
def carregar_csv(caminho: Path) -> pd.DataFrame:
    caminho = Path(caminho)
    if not caminho.exists():
        sys.exit(f"❌ Arquivo não encontrado: {caminho}")

    with caminho.open("r", encoding="utf-8") as f:
        amostra = "".join(next(f) for _ in range(5))
    sep = csv.Sniffer().sniff(amostra, delimiters=";,").delimiter

    df = pd.read_csv(caminho, sep=sep, dtype=str).rename(
        columns=lambda c: c.lower().strip()
    )

    # mapeia nomes usuais
    ren = {}
    for c in df.columns:
        if "quest" in c:
            ren[c] = "questao"
        elif "resp" in c:
            ren[c] = "resposta"
        elif "cand" in c:
            ren[c] = "candidato"
    df.rename(columns=ren, inplace=True)

    try:
        df = df[["questao", "resposta", "candidato"]]
    except KeyError:
        sys.exit("❌ Cabeçalhos esperados: questao, resposta, candidato")

    df["questao"] = df["questao"].astype(int)
    df["resposta"] = df["resposta"].str.strip().str.upper()
    df["candidato"] = df["candidato"].str.strip()

    return df


# ╭───────────────────────────────╮
# │ 3. Verificar correspondência  │
# ╰───────────────────────────────╯
def checar_correspondencia(df_num: pd.DataFrame, df_code: pd.DataFrame) -> None:
    """Mostra estatísticas e possíveis problemas de mapeamento."""
    ids_num = df_num["candidato"].astype(int)
    ids_code = df_code["candidato"].apply(get_numeric_id)

    set_num, set_code = set(ids_num), set(ids_code)
    intersec = set_num & set_code

    faltam_no_code = sorted(set_num - set_code)
    faltam_no_num = sorted(set_code - set_num)

    print("\n── Verificação de correspondência de candidatos ──")
    print(f"IDs numéricos           : {len(set_num)}")
    print(f"IDs código              : {len(set_code)}")
    print(f"IDs presentes em ambos  : {len(intersec)}")
    print(f"IDs só no CSV numérico  : {len(faltam_no_code)}")
    print(f"IDs só no CSV de código : {len(faltam_no_num)}")

    # Validação do padrão dos códigos
    cod_invalidos = df_code.loc[~df_code["candidato"].str.match(PADRAO_CODE), "candidato"].unique()
    if len(cod_invalidos):
        print(f"\n⚠️  Códigos fora do padrão 010NNN01 detectados: {cod_invalidos[:10]}{' …' if len(cod_invalidos) > 10 else ''}")

    # Exibe alguns exemplos de mapeamento correto/inexistente
    exemplos = [1, 2, max(intersec) if intersec else 1]
    print("\nExemplos:")
    for n in exemplos:
        print(f"  {n:3d}  →  esperado '{numeric2code(n)}'  "
              f"| no_arquivo_numérico={'✓' if n in set_num else '✗'}  "
              f"| no_arquivo_codigo={'✓' if n in set_code else '✗'}")


# ╭───────────────────────────────╮
# │ 4. Comparação                 │
# ╰───────────────────────────────╯
def comparar(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df1 = df1.copy()
    df2 = df2.copy()

    df1["cand_id"] = df1["candidato"].apply(get_numeric_id)
    df2["cand_id"] = df2["candidato"].apply(get_numeric_id)

    df1 = df1.drop_duplicates(subset=["cand_id", "questao"])
    df2 = df2.drop_duplicates(subset=["cand_id", "questao"])

    merged = df1.merge(
        df2[["cand_id", "questao", "resposta"]],
        on=["cand_id", "questao"],
        how="inner",
        suffixes=("_cand", "_ofc"),
    )
    merged["correto"] = merged["resposta_cand"] == merged["resposta_ofc"]
    return merged


def relatorio(result: pd.DataFrame, tempo: float, mem_mb: float) -> None:
    total = len(result)
    corretos = result["correto"].sum()
    acc_global = 100 * corretos / total if total else 0

    print("\n─── RESULTADOS ─────────────────────────────────────────")
    print(f"Comparações realizadas : {total}")
    print(f"Respostas idênticas    : {corretos}")
    print(f"Acurácia global        : {acc_global:.2f}%")

    print("\nAcurácia por candidato (ID):")
    acc_por_cand = (
        result.groupby("cand_id")["correto"].mean().mul(100).round(2).sort_index()
    )
    if len(acc_por_cand):
        print(acc_por_cand.to_string())
    else:
        print("─ Nenhuma intersecção de candidato+questão ─")

    print("\n── Medições de desempenho ──")
    print(f"Tempo de processamento : {tempo:.3f} s")
    print(f"Pico de memória        : {mem_mb:.2f} MB")
    print("─────────────────────────────────────────────────────────")


# ╭───────────────────────────────╮
# │ 5. Programa principal         │
# ╰───────────────────────────────╯
def main():
    parser = argparse.ArgumentParser(
        description="Compara respostas de dois CSVs após verificar a correspondência de IDs."
    )
    parser.add_argument("--csv1", required=True, type=Path, help="respostas_candidatos.csv (numérico)")
    parser.add_argument("--csv2", required=True, type=Path, help="respostas_.csv (código 010NNN01)")
    args = parser.parse_args()

    tracemalloc.start()
    t0 = time.perf_counter()

    # Carrega
    df1 = carregar_csv(args.csv1)
    df2 = carregar_csv(args.csv2)

    # Identifica qual tem números simples e qual tem códigos 010NNN01
    if df1["candidato"].str.match(PADRAO_CODE).all():
        df_code, df_num = df1, df2
    else:
        df_num, df_code = df1, df2

    # 1) Checagem de correspondência
    checar_correspondencia(df_num, df_code)

    # 2) Comparação de respostas
    resultado = comparar(df1, df2)

    # 3) Desempenho
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    relatorio(resultado, tempo=t1 - t0, mem_mb=peak / (1024 * 1024))


if __name__ == "__main__":
    main()
