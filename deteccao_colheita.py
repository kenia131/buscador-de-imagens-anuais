"""
deteccao_colheita.py

Ferramentas para acelerar a interpretação visual do analista:

1. `serie_temporal_ndvi_medio`: calcula o NDVI médio (dentro do talhão)
   para cada ano da série, formando uma série temporal simples.
2. `sugerir_anos_candidatos`: a partir dessa série, sugere os anos com
   maior queda de NDVI em relação ao ano anterior — candidatos fortes a
   "ano de colheita" — para que o analista confirme visualmente em vez
   de precisar olhar as ~35 composições uma a uma.
3. `RegistroColheitas`: um pequeno gerenciador de planilha (CSV/pandas)
   para registrar, para cada talhão, o ano de colheita identificado,
   substituindo o preenchimento manual de planilha do fluxo original.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def serie_temporal_ndvi_medio(composicoes_por_ano):
    """
    Recebe um dicionário {ano: composicao_xarray (ou None)} e devolve um
    pandas.Series indexado por ano com o NDVI médio do talhão naquele
    ano (ignorando anos sem imagem válida / composicao None).
    """
    from composicao_anual import calcular_ndvi

    valores = {}
    for ano, composicao in composicoes_por_ano.items():
        if composicao is None:
            continue
        ndvi = calcular_ndvi(composicao)
        media = float(np.nanmean(ndvi.values))
        if np.isfinite(media):
            valores[ano] = media

    serie = pd.Series(valores).sort_index()
    serie.index.name = "ano"
    serie.name = "ndvi_medio"
    return serie


def sugerir_anos_candidatos(serie_ndvi, queda_minima=0.15, top_n=3):
    """
    Sugere até `top_n` anos como candidatos a colheita: aqueles em que o
    NDVI médio caiu pelo menos `queda_minima` (em valor absoluto de NDVI,
    ex.: 0.15) em relação ao ano anterior disponível na série.

    Retorna uma lista de dicionários ordenada da queda mais forte para a
    mais fraca: [{"ano": 2015, "queda_ndvi": 0.32, "ndvi_antes": 0.78,
    "ndvi_depois": 0.46}, ...]
    """
    candidatos = []
    anos = list(serie_ndvi.index)

    for i in range(1, len(anos)):
        ano_anterior, ano_atual = anos[i - 1], anos[i]
        ndvi_antes = serie_ndvi.loc[ano_anterior]
        ndvi_depois = serie_ndvi.loc[ano_atual]
        queda = ndvi_antes - ndvi_depois

        if queda >= queda_minima:
            candidatos.append({
                "ano": int(ano_atual),
                "queda_ndvi": round(float(queda), 3),
                "ndvi_antes": round(float(ndvi_antes), 3),
                "ndvi_depois": round(float(ndvi_depois), 3),
            })

    candidatos.sort(key=lambda c: c["queda_ndvi"], reverse=True)
    return candidatos[:top_n]


class RegistroColheitas:
    """
    Gerenciador simples da planilha de resultados (equivalente ao
    "preencher uma planilha com o id do talhão e o ano de colheita" do
    seu fluxo atual), salva incrementalmente em CSV para não perder
    progresso entre sessões.
    """

    COLUNAS = ["talhao_id", "ano_colheita", "confianca_visual", "observacoes"]

    def __init__(self, caminho_csv="colheitas_identificadas.csv"):
        self.caminho_csv = Path(caminho_csv)
        if self.caminho_csv.exists():
            self.df = pd.read_csv(self.caminho_csv)
        else:
            self.df = pd.DataFrame(columns=self.COLUNAS)

    def registrar(self, talhao_id, ano_colheita, confianca_visual="alta", observacoes=""):
        """
        Registra (ou atualiza, se o talhão já tiver um registro) o ano
        de colheita identificado para o talhão. Salva o CSV imediatamente.
        """
        nova_linha = {
            "talhao_id": talhao_id,
            "ano_colheita": ano_colheita,
            "confianca_visual": confianca_visual,
            "observacoes": observacoes,
        }

        if talhao_id in self.df["talhao_id"].values:
            idx = self.df.index[self.df["talhao_id"] == talhao_id][0]
            for chave, valor in nova_linha.items():
                self.df.at[idx, chave] = valor
        else:
            self.df = pd.concat([self.df, pd.DataFrame([nova_linha])], ignore_index=True)

        self.df.to_csv(self.caminho_csv, index=False)
        return self.df

    def ja_registrado(self, talhao_id):
        return talhao_id in self.df["talhao_id"].values

    def exportar_excel(self, caminho_xlsx="colheitas_identificadas.xlsx"):
        self.df.to_excel(caminho_xlsx, index=False)
        return caminho_xlsx
