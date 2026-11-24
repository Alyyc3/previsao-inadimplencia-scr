"""Funções para baixar o SCR.data (Banco Central) e montar o painel mensal."""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import requests

URL_BASE = "https://www.bcb.gov.br/pda/desig/scrdata_{ano}.zip"
RAIZ = Path(__file__).resolve().parents[1]
PASTA_RAW = RAIZ / "data" / "raw"
PASTA_PROCESSED = RAIZ / "data" / "processed"

CHAVES = ["data_base", "uf", "cliente", "modalidade"]
CHAVES_RECORTE = ["uf", "cliente", "modalidade"]
COLUNAS_VALOR = [
    "carteira_ativa",
    "carteira_inadimplencia",
    "carteira_a_vencer",
    "a_vencer_ate_90_dias",
    "vencido_de_15_ate_90_dias",
    "vencido_acima_de_90_dias",
    "ativo_problematico",
]


def baixar_ano(ano: int) -> Path:
    """Baixa o ZIP de um ano para data/raw, se ainda não existir."""
    PASTA_RAW.mkdir(parents=True, exist_ok=True)
    destino = PASTA_RAW / f"scrdata_{ano}.zip"
    if destino.exists():
        return destino

    url = URL_BASE.format(ano=ano)
    temporario = destino.with_suffix(".zip.part")
    print(f"Baixando {url} ...")
    try:
        with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"}) as resposta:
            resposta.raise_for_status()
            with open(temporario, "wb") as arquivo:
                for bloco in resposta.iter_content(chunk_size=1024 * 1024):
                    arquivo.write(bloco)
    except requests.RequestException as erro:
        temporario.unlink(missing_ok=True)
        raise RuntimeError(f"Falha ao baixar {url}: {erro}") from erro

    if not zipfile.is_zipfile(temporario):
        temporario.unlink(missing_ok=True)
        raise RuntimeError(f"O arquivo de {ano} não é um ZIP válido.")
    temporario.replace(destino)
    return destino


def _ler_csv(zf: zipfile.ZipFile, nome: str, **kwargs) -> pd.DataFrame:
    """Lê um CSV do ZIP no formato do SCR.data; tenta UTF-8 e depois Latin-1."""
    for codificacao in ("utf-8", "latin-1"):
        try:
            with zf.open(nome) as arquivo:
                return pd.read_csv(arquivo, sep=";", decimal=",", encoding=codificacao, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Não consegui decodificar {nome}.")


def agregar_mes(zf: zipfile.ZipFile, nome: str) -> pd.DataFrame:
    """Lê um mês só com as colunas necessárias e soma os valores por recorte."""
    tipos = {c: "category" for c in CHAVES} | {c: "float64" for c in COLUNAS_VALOR}
    df = _ler_csv(zf, nome, usecols=CHAVES + COLUNAS_VALOR, dtype=tipos)
    return df.groupby(CHAVES, observed=True, as_index=False)[COLUNAS_VALOR].sum()


def montar_painel(anos: list[int]) -> pd.DataFrame:
    """Monta o painel mensal agregado por UF, tipo de cliente e modalidade."""
    partes = []
    for ano in anos:
        caminho = baixar_ano(ano)
        with zipfile.ZipFile(caminho) as zf:
            nomes = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
            print(f"{ano}: {len(nomes)} arquivos mensais")
            for nome in nomes:
                try:
                    partes.append(agregar_mes(zf, nome))
                except ValueError as erro:
                    raise ValueError(f"Erro ao ler {nome}: {erro}") from erro

    painel = pd.concat(partes, ignore_index=True)
    painel["data_base"] = pd.to_datetime(painel["data_base"].astype(str))
    for coluna in CHAVES_RECORTE:
        painel[coluna] = painel[coluna].astype("category")
    return painel.sort_values(CHAVES_RECORTE + ["data_base"]).reset_index(drop=True)