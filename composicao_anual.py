"""
composicao_anual.py

Versão simplificada, SEM mascaramento de nuvem: para uma geometria
(talhão) e um ano, busca as cenas disponíveis no STAC do Microsoft
Planetary Computer e retorna diretamente a cena com a MENOR cobertura de
nuvem (`eo:cloud_cover`) daquele ano — uma única imagem, sem composição
por mediana. A visualização usa a mesma combinação de bandas do script
GEE original (NIR/SWIR1/RED), mas com uma janela de contraste FIXA
(min=0, max=0.5) em vez de um realce por percentil.

Fonte de dados: Microsoft Planetary Computer (STAC), gratuito, sem
necessidade de conta/login para busca nem leitura dos dados (acesso
anônimo via SAS token gerado na hora). A coleção `landsat-c2-l2` cobre de
forma contínua 1984 até hoje (Landsat 4/5/7/8/9), o que cobre bem
qualquer intervalo de anos do seu caso de uso.
"""

import numpy as np
import planetary_computer
import pystac_client
import rioxarray  # necessário para registrar o acessador `.rio` em xarray.DataArray
import stackstac

STAC_URL_PLANETARY_COMPUTER = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Nomes de asset (banda) no Planetary Computer
BANDAS_LANDSAT = {
    "blue": "blue",
    "green": "green",
    "red": "red",
    "nir": "nir08",
    "swir1": "swir16",
    "swir2": "swir22",
}

BANDAS_SENTINEL2 = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "nir": "B08",
    "swir1": "B11",
    "swir2": "B12",
}

# NOTA IMPORTANTE: o `stackstac` já aplica automaticamente o fator de
# escala/offset de cada banda (lido do metadado `raster:bands` que o
# Planetary Computer inclui nos itens STAC) durante o empilhamento. Ou
# seja, os valores que saem de `stackstac.stack(...)` já são reflectância
# de superfície (0-1), NÃO o DN inteiro bruto (que ficaria na casa dos
# milhares). Aplicar uma escala manual aqui em cima disso duplicaria a
# escala (foi exatamente o bug que fez tudo virar ~0 após o clip).
#
# Estas constantes ficam guardadas só para referência/documentação — NÃO
# use-as multiplicando os valores vindos do stackstac.
LANDSAT_ESCALA_REFERENCIA = 0.0000275
LANDSAT_OFFSET_REFERENCIA = -0.2
SENTINEL2_ESCALA_REFERENCIA = 1.0 / 10000.0

# Janela de contraste fixa pedida (em vez de realce por percentil)
CONTRASTE_MINIMO_PADRAO = 0.0
CONTRASTE_MAXIMO_PADRAO = 0.5


def _abrir_catalogo(stac_url=STAC_URL_PLANETARY_COMPUTER):
    """Abre o catálogo STAC já configurado para assinar (sign) os assets
    automaticamente — necessário para o Planetary Computer, que exige um
    token de acesso de curta duração para ler os arquivos (gerado na hora,
    sem necessidade de conta/login)."""
    return pystac_client.Client.open(stac_url, modifier=planetary_computer.sign_inplace)


def _epsg_utm_para_geometria(geometria_wgs84):
    """
    Estima o EPSG de uma projeção UTM apropriada para a geometria
    informada (usa a técnica padrão do GeoPandas, que escolhe o fuso UTM
    com base no centróide). É necessário informar essa CRS explicitamente
    ao `stackstac`, pois alguns itens do Planetary Computer (ex.: cenas
    Landsat) não trazem a projeção (`proj:epsg`) no nível do asset — só no
    nível do item — e sem isso o `stackstac` não consegue decidir uma CRS
    comum sozinho (erro "Cannot pick a common CRS").
    """
    import geopandas as gpd

    serie = gpd.GeoSeries([geometria_wgs84], crs="EPSG:4326")
    crs_utm = serie.estimate_utm_crs()
    return crs_utm.to_epsg()


def _colecao_stac(fonte):
    if fonte == "landsat":
        return "landsat-c2-l2"
    if fonte == "sentinel2":
        return "sentinel-2-l2a"
    raise ValueError("fonte deve ser 'landsat' ou 'sentinel2'")


def buscar_todas_as_cenas(geometria_wgs84, ano, fonte="landsat", mes_inicio=5, mes_fim=7):
    """Busca TODAS as cenas (sem filtro de nuvem) da coleção escolhida que
    intersectam a geometria no intervalo de datas informado."""
    catalogo = _abrir_catalogo()
    colecao = _colecao_stac(fonte)
    datetime_str = f"{ano}-{mes_inicio:02d}-01/{ano}-{mes_fim:02d}-28"
    busca = catalogo.search(
        collections=[colecao],
        intersects=geometria_wgs84.__geo_interface__,
        datetime=datetime_str,
    )
    return list(busca.items())


def selecionar_cena_menos_nublada(geometria_wgs84, ano, fonte="landsat", mes_inicio=1, mes_fim=12):
    """
    Busca todas as cenas do ano e retorna o item (pystac.Item) com a
    MENOR cobertura de nuvem (`eo:cloud_cover`). Retorna None se nenhuma
    cena for encontrada para o período/área.

    Para Landsat, prioriza cenas Tier 1 (`landsat:collection_category`
    == "T1") sobre Tier 2: T1 passou por correção geométrica de precisão
    com pontos de controle no solo (geolocalização confiável, poucos
    metros de erro); T2 só tem correção sistemática/orbital, que pode
    ter erro de geolocalização de centenas de metros a QUILÔMETROS —
    comum em cenas antigas do Landsat 4/5 (ex.: anos 1990) em regiões com
    pouca cobertura de pontos de controle. Só cai para T2 se não existir
    NENHUMA cena T1 disponível naquele ano/área.
    """
    itens = buscar_todas_as_cenas(geometria_wgs84, ano, fonte, mes_inicio, mes_fim)
    if not itens:
        return None

    if fonte == "landsat":
        itens_tier1 = [i for i in itens if i.properties.get("landsat:collection_category") == "T1"]
        if itens_tier1:
            itens = itens_tier1

    return min(itens, key=lambda item: item.properties.get("eo:cloud_cover", 100))


def _carregar_cena_unica(item, bandas, resolucao, geometria_buffer_wgs84):
    """Carrega as bandas de um único item STAC como um xarray (banda, y, x),
    já recortado (bounds) na área de interesse, sem nenhuma máscara."""
    bounds = geometria_buffer_wgs84.bounds  # (minx, miny, maxx, maxy) em WGS84
    epsg_destino = _epsg_utm_para_geometria(geometria_buffer_wgs84)
    pilha = stackstac.stack(
        [item],
        assets=list(bandas.values()),
        resolution=resolucao,
        bounds_latlon=bounds,
        epsg=epsg_destino,
        dtype="float64",
        fill_value=np.nan,
    )
    inverso = {v: k for k, v in bandas.items()}
    pilha = pilha.assign_coords(band=[inverso[b] for b in pilha.band.values])
    imagem = pilha.isel(time=0)
    imagem = imagem.rio.write_crs(f"EPSG:{epsg_destino}")
    return imagem


def composicao_anual(
    geometria_wgs84,
    ano,
    geometria_buffer_wgs84=None,
    fonte="landsat",
    resolucao=30,
    mes_inicio=1,
    mes_fim=12,
    bandas_desejadas=("red", "nir", "swir1"),
    **kwargs,  # aceita e ignora parâmetros antigos (ex.: nuvem_max) por compatibilidade
):
    """
    Retorna a cena com MENOR cobertura de nuvem do ano para a geometria
    informada — SEM nenhuma máscara de nuvem aplicada (a seleção é feita
    apenas escolhendo a cena mais "limpa" disponível, não mascarando
    pixel a pixel).

    fonte: 'landsat' (recomendado, cobre 1984-hoje, resolução 30 m) ou
           'sentinel2' (só a partir de 2015-2016, resolução 10 m).

    `bandas_desejadas`: quais bandas baixar (das disponíveis em
    `BANDAS_LANDSAT`/`BANDAS_SENTINEL2`). Por padrão baixa só
    ("red", "nir", "swir1") — o suficiente para a falsa cor NIR/SWIR1/RED
    e o cálculo de NDVI — em vez das 6 bandas completas, o que reduz bem
    o volume de dados transferido e acelera o processamento. Passe
    `bandas_desejadas=("blue","green","red","nir","swir1","swir2")` se
    precisar de todas.

    Retorna um xarray.DataArray com dimensão 'band' contendo as bandas
    pedidas — já em reflectância de superfície, recortado na geometria
    (buffer); ou None se nenhuma cena for encontrada para o ano/área.
    """
    if geometria_buffer_wgs84 is None:
        geometria_buffer_wgs84 = geometria_wgs84

    item = selecionar_cena_menos_nublada(geometria_wgs84, ano, fonte, mes_inicio, mes_fim)
    if item is None:
        return None

    bandas_disponiveis = BANDAS_LANDSAT if fonte == "landsat" else BANDAS_SENTINEL2
    bandas = {nome: asset for nome, asset in bandas_disponiveis.items() if nome in bandas_desejadas}
    imagem = _carregar_cena_unica(item, bandas, resolucao, geometria_buffer_wgs84)

    # IMPORTANTE: o comportamento de escala automática do stackstac (via
    # metadado `raster:bands`) NÃO é uniforme entre coleções. Confirmado
    # empiricamente:
    #   - Landsat (landsat-c2-l2): os assets já trazem `raster:bands` com
    #     escala/offset, e o stackstac aplica isso sozinho — os valores
    #     já saem como reflectância 0-1. NÃO reescalar aqui (fizemos esse
    #     bug antes: reescalar de novo levava tudo para ~0).
    #   - Sentinel-2 (sentinel-2-l2a): os assets NÃO trazem essa mesma
    #     metadata por asset no Planetary Computer, então o stackstac
    #     devolve o DN bruto (tipicamente 0-10000+) sem aplicar nada —
    #     por isso é preciso dividir por 10000 manualmente aqui. Sem essa
    #     divisão, o `.clip(max=1)` satura tudo em exatamente 1.0 (foi
    #     esse o bug reportado).
    if fonte == "sentinel2":
        imagem = imagem / 10000.0

    imagem = imagem.clip(min=0, max=1)  # garante a faixa física válida de reflectância

    # IMPORTANTE: o stackstac devolve um array PREGUIÇOSO (dask) — os pixels
    # só são de fato lidos (download da cena via HTTP no Azure Blob Storage)
    # quando algo acessa `.values`. Sem materializar aqui, cada vez que a
    # imagem é plotada (inclusive ao reabrir do cache em disco) a leitura de
    # rede é disparada DE NOVO — é isso que causa a demora de minutos por
    # talhão a cada replot. `.compute()` força a leitura UMA única vez, logo
    # após montar a composição, e devolve os pixels já em memória (numpy).
    imagem = imagem.compute()

    imagem.attrs["ano"] = ano
    imagem.attrs["fonte"] = fonte
    imagem.attrs["n_cenas"] = 1
    imagem.attrs["item_id"] = item.id
    imagem.attrs["nuvem_pct"] = item.properties.get("eo:cloud_cover")
    imagem.attrs["tier"] = item.properties.get("landsat:collection_category")  # T1/T2 (só Landsat); None p/ Sentinel-2
    return imagem


def calcular_ndvi(composicao):
    """NDVI = (NIR - RED) / (NIR + RED), a partir da cena selecionada."""
    nir = composicao.sel(band="nir")
    red = composicao.sel(band="red")
    return (nir - red) / (nir + red)


def aplicar_janela_fixa(
    composicao,
    bandas=("nir", "swir1", "red"),
    minimo=CONTRASTE_MINIMO_PADRAO,
    maximo=CONTRASTE_MAXIMO_PADRAO,
):
    """
    Converte a composição para uma imagem RGB (uint8) usando uma janela
    de contraste FIXA — todo valor de reflectância <= `minimo` vira 0,
    todo valor >= `maximo` vira 255, com escala linear entre os dois.
    Equivalente ao `vis = {bands: [...], min: 0, max: 0.5}` do script GEE
    original, em vez de um realce por percentil calculado por imagem.
    """
    canais = []
    for banda in bandas:
        dado = composicao.sel(band=banda).values.astype("float64")
        esticado = np.clip((dado - minimo) / (maximo - minimo), 0, 1)
        esticado = np.nan_to_num(esticado, nan=0.0)
        canais.append((esticado * 255).astype("uint8"))
    return np.dstack(canais)


# Alias para compatibilidade com o nome usado em versões anteriores do notebook
aplicar_contraste = aplicar_janela_fixa