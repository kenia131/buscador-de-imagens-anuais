"""
dashboard_imagens.py

Dashboard Streamlit (rodar com `streamlit run dashboard_imagens.py`) para
visualizar o histórico anual de imagens de satélite (falsa cor
NIR/SWIR1/RED) de um talhão, sem precisar de banco de dados.

Fluxo: o usuário informa a geometria (GeoJSON, WKT ou arquivo), o app
busca as imagens no Microsoft Planetary Computer (STAC, gratuito e
anônimo) e mostra a grade de anos lado a lado, com o contorno do talhão
sobreposto. Não há nenhuma persistência em SQL/NoSQL — o único "cache" é
um diretório local de arquivos .pkl, para não repetir buscas já feitas.
"""

import concurrent.futures
import io
import pickle
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # evita tentativa de abrir janela grafica no servidor
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from composicao_anual import aplicar_janela_fixa, composicao_anual
from deteccao_colheita import serie_temporal_ndvi_medio, sugerir_anos_candidatos
from geometria_utils import (
    buffer_metros,
    carregar_arquivo_geometria,
    carregar_geojson_texto,
    carregar_wkt_texto,
)

# --- CONFIGURAÇÃO DA PÁGINA ---

st.set_page_config(page_title="Histórico de imagens do talhão", layout="wide")
st.title("🛰️ Histórico anual de imagens de satélite por talhão")
st.caption(
    "Fonte: Microsoft Planetary Computer (STAC) — acesso gratuito e anônimo, "
    "sem necessidade de conta, chave de API ou banco de dados."
)

CACHE_DIR = Path("cache_composicoes")
CACHE_DIR.mkdir(exist_ok=True)


# --- FUNÇÕES DE PROCESSAMENTO (mesma lógica usada no notebook) ---

def caminho_cache(talhao_id, ano, fonte):
    nome = f"{talhao_id}_{ano}_{fonte}.pkl".replace("/", "_").replace(" ", "_")
    return CACHE_DIR / nome


def fonte_e_resolucao_do_ano(ano, ano_transicao_sentinel):
    """Sentinel-2 (10 m) para os anos recentes; Landsat (30 m) para os
    mais antigos, onde o Sentinel-2 ainda não existia."""
    if ano >= ano_transicao_sentinel:
        return "sentinel2", 10
    return "landsat", 30


def processar_talhao(
    talhao_id,
    geometria,
    ano_inicio,
    ano_fim,
    ano_transicao_sentinel,
    buffer_m,
    max_workers=6,
    usar_cache=True,
    callback_progresso=None,
):
    """
    Monta a composição anual de cada ano do intervalo, buscando em
    paralelo os anos que ainda não estão em cache. Retorna
    {ano: composicao_xarray|None}.
    """
    geometria_buffer = buffer_metros(geometria, buffer_m)
    composicoes = {}
    anos_para_buscar = []

    for ano in range(ano_inicio, ano_fim + 1):
        fonte, _ = fonte_e_resolucao_do_ano(ano, ano_transicao_sentinel)
        arquivo_cache = caminho_cache(talhao_id, ano, fonte)
        if usar_cache and arquivo_cache.exists():
            with open(arquivo_cache, "rb") as f:
                composicoes[ano] = pickle.load(f)
        else:
            anos_para_buscar.append(ano)

    total = ano_fim - ano_inicio + 1
    concluidos = total - len(anos_para_buscar)
    erros = []
    if callback_progresso:
        callback_progresso(concluidos, total)

    def _buscar_um_ano(ano):
        fonte, resolucao = fonte_e_resolucao_do_ano(ano, ano_transicao_sentinel)
        try:
            composicao = composicao_anual(
                geometria,
                ano,
                geometria_buffer_wgs84=geometria_buffer,
                fonte=fonte,
                resolucao=resolucao,
            )
        except Exception as erro:
            composicao = None
            erros.append(f"{ano} ({fonte}): {erro}")
        return ano, fonte, composicao

    if anos_para_buscar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for ano, fonte, composicao in executor.map(_buscar_um_ano, anos_para_buscar):
                composicoes[ano] = composicao
                if usar_cache:
                    with open(caminho_cache(talhao_id, ano, fonte), "wb") as f:
                        pickle.dump(composicao, f)
                concluidos += 1
                if callback_progresso:
                    callback_progresso(concluidos, total)

    return composicoes, erros


def processar_talhao_incremental(
    talhao_id,
    geometria,
    ano_inicio,
    ano_fim,
    ano_transicao_sentinel,
    buffer_m,
    max_workers=6,
    usar_cache=True,
):
    """
    Gerador que produz (ano, fonte, composicao, erro) assim que cada ano
    fica pronto — anos já em cache são entregues imediatamente (na ordem
    cronológica); anos que precisam ser buscados são processados em
    paralelo e entregues na ordem em que TERMINAM de baixar (via
    `as_completed`, não `map`), permitindo mostrar cada imagem assim que
    estiver pronta, sem esperar o lote inteiro terminar.
    """
    geometria_buffer = buffer_metros(geometria, buffer_m)
    anos_para_buscar = []

    for ano in range(ano_inicio, ano_fim + 1):
        fonte, _ = fonte_e_resolucao_do_ano(ano, ano_transicao_sentinel)
        arquivo_cache = caminho_cache(talhao_id, ano, fonte)
        if usar_cache and arquivo_cache.exists():
            with open(arquivo_cache, "rb") as f:
                composicao = pickle.load(f)
            yield ano, fonte, composicao, None
        else:
            anos_para_buscar.append(ano)

    if not anos_para_buscar:
        return

    def _buscar_um_ano(ano):
        fonte, resolucao = fonte_e_resolucao_do_ano(ano, ano_transicao_sentinel)
        try:
            composicao = composicao_anual(
                geometria, ano,
                geometria_buffer_wgs84=geometria_buffer,
                fonte=fonte, resolucao=resolucao,
            )
            erro = None
        except Exception as e:
            composicao = None
            erro = f"{ano} ({fonte}): {e}"
        return ano, fonte, composicao, erro

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_buscar_um_ano, ano): ano for ano in anos_para_buscar}
        for future in concurrent.futures.as_completed(futures):
            ano, fonte, composicao, erro = future.result()
            if usar_cache:
                with open(caminho_cache(talhao_id, ano, fonte), "wb") as f:
                    pickle.dump(composicao, f)
            yield ano, fonte, composicao, erro


def plotar_imagem_individual(
    composicao,
    geometria,
    bandas=("nir", "swir1", "red"),
    minimo=0.0,
    maximo=0.5,
    destaque_candidato=False,
    cache_crs=None,
):
    """
    Plota UMA única imagem (não a grade inteira) — usada no modo
    incremental, onde cada ano vira uma figura pequena e independente,
    exibida assim que fica pronta. `cache_crs` (um dict compartilhado
    entre chamadas) evita reprojetar o talhão do zero a cada imagem.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    rgb = aplicar_janela_fixa(composicao, bandas=bandas, minimo=minimo, maximo=maximo)

    dx = abs(float(composicao.x[1] - composicao.x[0]))
    dy = abs(float(composicao.y[0] - composicao.y[1]))
    extent = [
        float(composicao.x.min()) - dx / 2, float(composicao.x.max()) + dx / 2,
        float(composicao.y.min()) - dy / 2, float(composicao.y.max()) + dy / 2,
    ]
    ax.imshow(rgb, extent=extent, origin="upper")

    crs_imagem = str(composicao.rio.crs)
    if cache_crs is not None and crs_imagem in cache_crs:
        gdf_talhao = cache_crs[crs_imagem]
    else:
        gdf_talhao = gpd.GeoSeries([geometria], crs="EPSG:4326").to_crs(composicao.rio.crs)
        if cache_crs is not None:
            cache_crs[crs_imagem] = gdf_talhao
    gdf_talhao.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.5)

    fonte = composicao.attrs.get("fonte", "?")
    nuvem_pct = composicao.attrs.get("nuvem_pct")
    tier = composicao.attrs.get("tier")
    ano = composicao.attrs.get("ano")
    #info_nuvem = f", {nuvem_pct:.0f}% nuvem" if nuvem_pct is not None else ""
    aviso_tier = " ⚠️T2" if tier == "T2" else ""
    aviso_candidato = " 🔴" if destaque_candidato else ""
    ax.set_title(f"{ano} ({fonte}){aviso_tier}{aviso_candidato}", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    if destaque_candidato:
        for borda in ax.spines.values():
            borda.set_edgecolor("red")
            borda.set_linewidth(3)

    plt.tight_layout()
    return fig


def plotar_grade_anual(
    talhao_id,
    geometria,
    composicoes,
    bandas=("nir", "swir1", "red"),
    minimo=0.0,
    maximo=0.5,
    ncols=6,
    anos_candidatos=None,
):
    """
    Uma imagem por ano, grade de `ncols` colunas, com o contorno do
    talhão sobreposto (borda preta, interior transparente). Anos
    presentes em `anos_candidatos` (sugeridos pela queda de NDVI como
    prováveis anos de colheita) têm a borda do subplot destacada em
    vermelho, para chamar atenção na grade.
    """
    anos_com_imagem = sorted([ano for ano, comp in composicoes.items() if comp is not None])
    if not anos_com_imagem:
        return None

    anos_candidatos = set(anos_candidatos or [])

    n = len(anos_com_imagem)
    nrows = int(np.ceil(n / ncols))
    fig, eixos = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

    # Reprojeta o talhão apenas uma vez por CRS distinto
    talhao_por_crs = {}

    for i, ano in enumerate(anos_com_imagem):
        linha, coluna = divmod(i, ncols)
        ax = eixos[linha][coluna]

        composicao = composicoes[ano]
        rgb = aplicar_janela_fixa(composicao, bandas=bandas, minimo=minimo, maximo=maximo)

        # x/y do stackstac são o CENTRO de cada pixel — soma/subtrai meio
        # pixel para chegar na borda real (evita a imagem aparecer
        # deslocada em relação ao contorno do talhão).
        dx = abs(float(composicao.x[1] - composicao.x[0]))
        dy = abs(float(composicao.y[0] - composicao.y[1]))
        extent = [
            float(composicao.x.min()) - dx / 2, float(composicao.x.max()) + dx / 2,
            float(composicao.y.min()) - dy / 2, float(composicao.y.max()) + dy / 2,
        ]
        ax.imshow(rgb, extent=extent, origin="upper")

        crs_imagem = str(composicao.rio.crs)
        if crs_imagem not in talhao_por_crs:
            talhao_por_crs[crs_imagem] = gpd.GeoSeries([geometria], crs="EPSG:4326").to_crs(composicao.rio.crs)
        talhao_por_crs[crs_imagem].plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.5)

        fonte = composicao.attrs.get("fonte", "?")
        nuvem_pct = composicao.attrs.get("nuvem_pct")
        tier = composicao.attrs.get("tier")
        info_nuvem = f", {nuvem_pct:.0f}% nuvem" if nuvem_pct is not None else ""
        aviso_tier = " ⚠️ T2" if tier == "T2" else ""
        eh_candidato = ano in anos_candidatos
        aviso_candidato = " 🔴 candidato a colheita" if eh_candidato else ""
        ax.set_title(f"{ano} ({fonte}{info_nuvem}){aviso_tier}{aviso_candidato}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        if eh_candidato:
            for borda in ax.spines.values():
                borda.set_edgecolor("red")
                borda.set_linewidth(4)

    for j in range(n, nrows * ncols):
        linha, coluna = divmod(j, ncols)
        eixos[linha][coluna].axis("off")

    fig.suptitle(f"Talhão {talhao_id} — falsa cor NIR/SWIR1/RED (janela {minimo}-{maximo})", fontsize=14)
    plt.tight_layout()
    return fig


# --- ESTADO DA SESSÃO ---

if "composicoes" not in st.session_state:
    st.session_state.composicoes = None
    st.session_state.talhao_id = None
    st.session_state.geometria_processada = None
    st.session_state.anos_candidatos = []
    st.session_state.detalhes_candidatos = []
    st.session_state.figura_cache_params = None  # (talhao_id, ncols, minimo, maximo) do último render
    st.session_state.figura_cache_fig = None
    st.session_state.figura_cache_png = None
    st.session_state.modo_usado = None  # "incremental" ou "classico" — controla a Seção 4


# --- SEÇÃO 1: GEOMETRIA ---

st.header("1. Geometria do talhão")
modo_entrada = st.radio(
    "Como enviar a geometria?",
    ["Colar GeoJSON", "Colar WKT", "Enviar arquivo (.geojson/.kml/.wkt)"],
    horizontal=True,
)

geometria = None
talhao_id = "talhao_dashboard"

if modo_entrada == "Colar GeoJSON":
    texto_geojson = st.text_area(
        "GeoJSON (FeatureCollection, Feature ou geometria)",
        height=150,
        placeholder='{"type": "Polygon", "coordinates": [[[...]]]}',
    )
    if texto_geojson.strip():
        try:
            talhoes = carregar_geojson_texto(texto_geojson)
            talhao_id, geometria = talhoes[0]
        except Exception as e:
            st.error(f"Erro ao interpretar o GeoJSON: {e}")

elif modo_entrada == "Colar WKT":
    texto_wkt = st.text_area("WKT", height=150, placeholder="MULTIPOLYGON Z (((...)))")
    crs_origem = st.text_input(
        "CRS de origem do WKT (obrigatório — o WKT não guarda a projeção)",
        value="EPSG:5880",
        help="Ex.: EPSG:5880 (SIRGAS 2000/Brasil Polyconic), EPSG:4326 se já estiver em graus decimais.",
    )
    if texto_wkt.strip():
        try:
            talhoes = carregar_wkt_texto(texto_wkt, crs_origem=crs_origem)
            talhao_id, geometria = talhoes[0]
        except Exception as e:
            st.error(f"Erro ao interpretar o WKT: {e}")

else:
    arquivo = st.file_uploader("Arquivo", type=["geojson", "json", "kml", "wkt"])
    crs_origem_arquivo = st.text_input(
        "CRS de origem (só necessário para arquivos .wkt)", value="EPSG:5880"
    )
    if arquivo is not None:
        caminho_tmp = Path(f"/tmp/{arquivo.name}")
        caminho_tmp.write_bytes(arquivo.getvalue())
        try:
            talhoes = carregar_arquivo_geometria(caminho_tmp, crs_origem=crs_origem_arquivo)
            talhao_id, geometria = talhoes[0]
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo: {e}")

if geometria is not None:
    st.success(f"Geometria carregada: {geometria.geom_type}, bounds={tuple(round(b, 5) for b in geometria.bounds)}")


# --- SEÇÃO 2: PARÂMETROS ---

st.header("2. Parâmetros")
col1, col2, col3 = st.columns(3)
with col1:
    ano_inicio = st.number_input("Ano inicial", min_value=1984, max_value=2026, value=2015)
with col2:
    ano_fim = st.number_input("Ano final", min_value=1984, max_value=2026, value=2026)
with col3:
    ano_transicao_sentinel = st.number_input(
        "Transição p/ Sentinel-2", min_value=2015, max_value=2026, value=2016,
        help="Anos >= este valor usam Sentinel-2 (10 m); anos anteriores usam Landsat (30 m).",
    )

col4, col5, col6 = st.columns(3)
with col4:
    buffer_valor = st.number_input("Buffer de contexto (m)", min_value=0, value=60)
with col5:
    ncols = st.number_input("Colunas na grade", min_value=1, max_value=10, value=6)
with col6:
    max_workers = st.number_input("Downloads em paralelo", min_value=1, max_value=12, value=6)

col7, col8 = st.columns(2)
with col7:
    minimo_contraste = st.number_input("Contraste — mínimo", value=0.0, step=0.05, format="%.2f")
with col8:
    maximo_contraste = st.number_input("Contraste — máximo", value=0.5, step=0.05, format="%.2f")

st.subheader("Detecção automática de colheita (queda de NDVI)")
col9, col10, col11 = st.columns(3)
with col9:
    detectar_candidatos = st.checkbox("Sugerir anos candidatos", value=True)
with col10:
    queda_minima_ndvi = st.number_input(
        "Queda mínima de NDVI", min_value=0.0, max_value=1.0, value=0.15, step=0.05, format="%.2f",
        help="Anos em que o NDVI médio caiu pelo menos esse valor em relação ao ano anterior "
             "são sugeridos como candidatos a colheita.",
    )
with col11:
    top_n_candidatos = st.number_input("Máximo de candidatos", min_value=1, max_value=10, value=3)

st.subheader("Modo de exibição")
modo_exibicao = st.radio(
    "Como mostrar as imagens?",
    ["Incremental (uma por vez, assim que fica pronta)", "Grade única (clássica, espera tudo terminar)"],
    horizontal=True,
    help="O modo incremental mostra cada ano assim que ele termina de baixar, sem esperar "
         "o lote inteiro — recomendado para buscas com muitos anos.",
)
modo_incremental = modo_exibicao.startswith("Incremental")


# --- SEÇÃO 3: PROCESSAR ---

st.header("3. Buscar imagens")

if st.button("🚀 Buscar/atualizar imagens", type="primary", use_container_width=True, disabled=(geometria is None)):
    st.session_state.talhao_id = talhao_id
    st.session_state.geometria_processada = geometria
    st.session_state.figura_cache_params = None  # invalida o cache de renderização (novos dados)

    if modo_incremental:
        st.session_state.modo_usado = "incremental"

        anos_totais = list(range(int(ano_inicio), int(ano_fim) + 1))
        n_total = len(anos_totais)
        ncols_int = int(ncols)

        st.subheader("Progresso")
        barra = st.progress(0.0, text="Iniciando...")

        st.subheader("Imagens (aparecem conforme ficam prontas)")
        placeholders = {}
        for i, ano in enumerate(anos_totais):
            coluna_idx = i % ncols_int
            if coluna_idx == 0:
                cols_atuais = st.columns(ncols_int)
            placeholders[ano] = cols_atuais[coluna_idx].empty()
            placeholders[ano].caption(f"{ano} — na fila...")

        composicoes = {}
        erros = []
        concluidos = 0
        cache_crs = {}  # evita reprojetar o talhão do zero a cada imagem

        for ano, fonte, composicao, erro in processar_talhao_incremental(
            talhao_id, geometria,
            ano_inicio=int(ano_inicio), ano_fim=int(ano_fim),
            ano_transicao_sentinel=int(ano_transicao_sentinel),
            buffer_m=buffer_valor, max_workers=int(max_workers),
        ):
            composicoes[ano] = composicao
            if erro:
                erros.append(erro)

            if composicao is not None:
                fig = plotar_imagem_individual(
                    composicao, geometria,
                    minimo=minimo_contraste, maximo=maximo_contraste,
                    cache_crs=cache_crs,
                )
                placeholders[ano].pyplot(fig)
                plt.close(fig)  # libera memória — importante para lotes grandes (200+ imagens)
            else:
                placeholders[ano].warning(f"{ano}: sem imagem")

            concluidos += 1
            barra.progress(concluidos / n_total, text=f"{concluidos}/{n_total} anos processados")

        barra.empty()
        st.session_state.composicoes = composicoes

        anos_ok = sorted([a for a, c in composicoes.items() if c is not None])
        anos_falha = sorted([a for a, c in composicoes.items() if c is None])
        st.success(f"{len(anos_ok)} de {n_total} ano(s) com imagem encontrada.")
        if anos_falha:
            st.warning(f"Sem imagem para: {anos_falha}")
        if erros:
            with st.expander(f"{len(erros)} erro(s) durante o processamento"):
                for msg in erros:
                    st.text(msg)

        # --- Detecção automática de colheita via queda de NDVI ---
        # (só é possível depois que TODOS os anos terminaram, já que compara
        # ano a ano; os placeholders já exibidos são atualizados no lugar
        # com a borda vermelha, sem precisar redesenhar a grade inteira)
        if detectar_candidatos:
            serie_ndvi = serie_temporal_ndvi_medio(composicoes)
            candidatos = sugerir_anos_candidatos(
                serie_ndvi, queda_minima=queda_minima_ndvi, top_n=int(top_n_candidatos)
            )
            st.session_state.anos_candidatos = [c["ano"] for c in candidatos]
            st.session_state.detalhes_candidatos = candidatos

            for c in candidatos:
                composicao_candidata = composicoes.get(c["ano"])
                if composicao_candidata is not None:
                    fig = plotar_imagem_individual(
                        composicao_candidata, geometria,
                        minimo=minimo_contraste, maximo=maximo_contraste,
                        destaque_candidato=True, cache_crs=cache_crs,
                    )
                    placeholders[c["ano"]].pyplot(fig)
                    plt.close(fig)

            if candidatos:
                st.info(
                    "🔴 Candidatos automáticos a ano de colheita (destacados em vermelho acima): "
                    + "; ".join(
                        f"**{c['ano']}** (NDVI caiu de {c['ndvi_antes']} para {c['ndvi_depois']}, "
                        f"queda de {c['queda_ndvi']})"
                        for c in candidatos
                    )
                )
            else:
                st.caption("Nenhuma queda de NDVI acima do limiar foi encontrada — revise a grade manualmente.")
        else:
            st.session_state.anos_candidatos = []
            st.session_state.detalhes_candidatos = []

    else:
        st.session_state.modo_usado = "classico"
        barra = st.progress(0.0, text="Iniciando...")

        def _atualizar_barra(concluidos, total):
            barra.progress(concluidos / total, text=f"{concluidos}/{total} anos processados")

        with st.spinner("Buscando imagens no Planetary Computer..."):
            composicoes, erros = processar_talhao(
                talhao_id,
                geometria,
                ano_inicio=int(ano_inicio),
                ano_fim=int(ano_fim),
                ano_transicao_sentinel=int(ano_transicao_sentinel),
                buffer_m=buffer_valor,
                max_workers=int(max_workers),
                callback_progresso=_atualizar_barra,
            )

        barra.empty()

        st.session_state.composicoes = composicoes

        anos_ok = sorted([a for a, c in composicoes.items() if c is not None])
        anos_falha = sorted([a for a, c in composicoes.items() if c is None])

        st.success(f"{len(anos_ok)} de {int(ano_fim) - int(ano_inicio) + 1} ano(s) com imagem encontrada.")
        if anos_falha:
            st.warning(f"Sem imagem para: {anos_falha}")
        if erros:
            with st.expander(f"{len(erros)} erro(s) durante o processamento"):
                for msg in erros:
                    st.text(msg)

        # --- Detecção automática de colheita via queda de NDVI ---
        if detectar_candidatos:
            serie_ndvi = serie_temporal_ndvi_medio(composicoes)
            candidatos = sugerir_anos_candidatos(
                serie_ndvi, queda_minima=queda_minima_ndvi, top_n=int(top_n_candidatos)
            )
            st.session_state.anos_candidatos = [c["ano"] for c in candidatos]
            st.session_state.detalhes_candidatos = candidatos

            if candidatos:
                st.info(
                    "🔴 Candidatos automáticos a ano de colheita (maior queda de NDVI primeiro): "
                    + "; ".join(
                        f"**{c['ano']}** (NDVI caiu de {c['ndvi_antes']} para {c['ndvi_depois']}, "
                        f"queda de {c['queda_ndvi']})"
                        for c in candidatos
                    )
                )
            else:
                st.caption("Nenhuma queda de NDVI acima do limiar foi encontrada — revise a grade manualmente.")
        else:
            st.session_state.anos_candidatos = []
            st.session_state.detalhes_candidatos = []


# --- SEÇÃO 4: RESULTADO (só no modo clássico — no incremental, as imagens já
# foram mostradas ao vivo durante o processamento na Seção 3) ---

if st.session_state.modo_usado == "classico" and st.session_state.composicoes is not None:
    st.header("4. Grade de imagens")
    if st.session_state.anos_candidatos:
        st.caption("🔴 Borda vermelha grossa = ano candidato a colheita (queda de NDVI)")

    parametros_atuais = (st.session_state.talhao_id, int(ncols), minimo_contraste, maximo_contraste)

    if parametros_atuais != st.session_state.figura_cache_params:
        # Algo mudou (ou é o primeiro render após uma busca nova) -> gera de fato
        fig = plotar_grade_anual(
            st.session_state.talhao_id,
            st.session_state.geometria_processada,
            st.session_state.composicoes,
            ncols=int(ncols),
            minimo=minimo_contraste,
            maximo=maximo_contraste,
            anos_candidatos=st.session_state.anos_candidatos,
        )
        st.session_state.figura_cache_params = parametros_atuais
        st.session_state.figura_cache_fig = fig
        if fig is not None:
            buffer_png = io.BytesIO()
            fig.savefig(buffer_png, format="png", dpi=150, bbox_inches="tight")
            st.session_state.figura_cache_png = buffer_png.getvalue()
        else:
            st.session_state.figura_cache_png = None
    else:
        # Nada mudou desde o último render (ex.: usuário só tocou em outro
        # widget da página) -> reaproveita a figura já pronta, sem recalcular
        fig = st.session_state.figura_cache_fig

    if fig is not None:
        st.pyplot(fig)
        st.download_button(
            "📥 Baixar grade como PNG",
            data=st.session_state.figura_cache_png,
            file_name=f"grade_{st.session_state.talhao_id}.png",
            mime="image/png",
        )
    else:
        st.error("Nenhuma imagem foi encontrada para o período informado.")