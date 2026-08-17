"""
geometria_utils.py

Funções para carregar a geometria de um talhão a partir de:
- texto colado em formato GeoJSON (como no script original do GEE);
- arquivo .geojson;
- arquivo .kml.

Sempre retorna a geometria em WGS84 (EPSG:4326), pronta para ser usada
nas buscas STAC (Planetary Computer / Brazil Data Cube).
"""

import hashlib
import json
import re
import warnings
from pathlib import Path

import fiona
import geopandas as gpd
import shapely
from shapely import wkt as shapely_wkt
from shapely.geometry import shape

fiona.drvsupport.supported_drivers['KML'] = 'rw'
fiona.drvsupport.supported_drivers['kml'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'


def _id_estavel_da_geometria(geometria):
    """
    Gera um identificador estável e determinístico a partir da própria
    geometria (hash das suas coordenadas — WKB).

    Usado como identificador de talhão quando nenhum campo de ID
    confiável foi informado/encontrado no arquivo. IMPORTANTE: antes,
    o fallback usado era o índice posicional (0, 1, 2...) da linha no
    GeoDataFrame — mas como cada upload de um único talhão sempre cria
    a linha de índice 0, TALHÕES DIFERENTES acabavam recebendo o MESMO
    identificador "0", fazendo o cache em disco (nomeado
    "{talhao_id}_{ano}_{fonte}.pkl") reaproveitar por engano as imagens
    de um talhão completamente diferente carregado anteriormente.

    Com o hash da geometria: geometrias diferentes sempre geram IDs
    diferentes (sem colisão de cache entre talhões), e a MESMA geometria
    sempre gera o mesmo ID (o cache legítimo entre sessões continua
    funcionando normalmente).
    """
    return "geom_" + hashlib.md5(geometria.wkb).hexdigest()[:12]


def _validar_e_reparar_geometria(geometria, identificador=None):
    """
    Verifica se a geometria é válida (sem auto-interseções etc.) e, se
    não for, tenta reparar automaticamente com `buffer(0)` — a técnica
    padrão para esse tipo de problema.

    Isso é importante porque uma geometria inválida pode causar erros
    obscuros mais adiante (ex.: `GEOSException: getY called on empty
    Point` ao calcular o centróide para o buffer de contexto), em vez de
    um erro claro no momento do carregamento — dependendo da versão do
    GEOS/Shapely instalada, calcular o centróide de um polígono
    auto-intersectante pode retornar uma geometria vazia em vez de
    lançar um erro.
    """
    if geometria is None or geometria.is_empty:
        raise ValueError(
            f"A geometria{f' do talhão {identificador}' if identificador else ''} "
            "está vazia."
        )

    if geometria.is_valid:
        return geometria

    geometria_reparada = geometria.buffer(0)

    if geometria_reparada.is_empty:
        raise ValueError(
            f"A geometria{f' do talhão {identificador}' if identificador else ''} "
            "está inválida (auto-interseção) e não foi possível repará-la "
            "automaticamente. Corrija o desenho em um software de GIS (ex.: "
            "'Fix geometries' no QGIS) antes de enviá-la."
        )

    if geometria.area > 0:
        diferenca_pct = abs(geometria.area - geometria_reparada.area) / geometria.area * 100
        if diferenca_pct > 1:
            warnings.warn(
                f"A geometria{f' do talhão {identificador}' if identificador else ''} "
                f"estava inválida (auto-interseção) e foi corrigida automaticamente, "
                f"mas isso alterou a área em {diferenca_pct:.1f}%. Vale a pena revisar "
                "o desenho original em um software de GIS para confirmar se está correto."
            )

    return geometria_reparada


def _geometria_unica_a_partir_de_gdf(gdf, id_campo=None):
    """
    Recebe um GeoDataFrame (uma ou mais feições) e retorna uma lista de
    tuplas (id_talhao, geometria_shapely) em WGS84, com a geometria já
    validada/reparada (ver `_validar_e_reparar_geometria`).

    Se `id_campo` for informado e existir no GeoDataFrame, usa seus
    valores como identificador do talhão; caso contrário, usa um
    identificador estável derivado da própria geometria (ver
    `_id_estavel_da_geometria`) — nunca o índice posicional, que causaria
    colisão de cache entre talhões diferentes.
    """
    if gdf.crs is None:
        raise ValueError(
            "O arquivo de geometria não possui projeção (CRS) definida. "
            "Defina a projeção (ex.: WGS84 / EPSG:4326) antes de usá-lo."
        )
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    resultado = []
    for idx, row in gdf.iterrows():
        identificador_temporario = row[id_campo] if (id_campo and id_campo in gdf.columns) else idx
        geometria_valida = _validar_e_reparar_geometria(row.geometry, identificador=identificador_temporario)

        if id_campo and id_campo in gdf.columns:
            talhao_id = row[id_campo]
        else:
            talhao_id = _id_estavel_da_geometria(geometria_valida)

        resultado.append((talhao_id, geometria_valida))
    return resultado


def _parece_sistema_projetado(coordenada_xy):
    """
    Heurística simples para detectar se uma coordenada claramente NÃO é
    WGS84 em graus decimais: latitude/longitude sempre ficam entre -180 e
    180. Coordenadas na casa das centenas de milhar ou milhões (como em
    UTM ou no SIRGAS 2000/Brasil Polyconic - EPSG:5880, que usa falso
    Easting = 5.000.000 e falso Northing = 10.000.000) só podem vir de um
    sistema de coordenadas PROJETADO.
    """
    x, y = coordenada_xy[0], coordenada_xy[1]
    return abs(x) > 1000 or abs(y) > 1000


def _extrair_epsg_do_membro_crs(dados_geojson):
    """
    Extrai o código EPSG do membro legado "crs" de um GeoJSON, se
    presente — ex.:
      "crs": {"type": "name", "properties":
              {"name": "urn:ogc:def:crs:EPSG::5880"}}

    Esse membro foi removido do padrão atual (RFC 7946, que assume
    sempre WGS84), mas softwares de GIS como QGIS e ArcGIS ainda o
    incluem ao exportar camadas em outras projeções. Ignorá-lo faria o
    código assumir erroneamente que coordenadas em metros (ex.: SIRGAS
    2000 / Brasil Polyconic) já estão em graus decimais.
    """
    if not isinstance(dados_geojson, dict):
        return None
    membro_crs = dados_geojson.get("crs")
    if not membro_crs:
        return None
    nome = membro_crs.get("properties", {}).get("name", "")
    match = re.search(r"EPSG::?(\d+)", nome, flags=re.IGNORECASE)
    return f"EPSG:{match.group(1)}" if match else None


def _primeira_coordenada(geometria):
    """Retorna a primeira coordenada (x, y) de uma geometria shapely,
    seja ela um Polygon, MultiPolygon, LineString etc."""
    if hasattr(geometria, "geoms"):
        geometria = geometria.geoms[0]
    anel = geometria.exterior if hasattr(geometria, "exterior") else geometria
    return list(anel.coords)[0]


def carregar_geojson_texto(texto_geojson, id_campo=None):
    """
    Equivalente ao `inputField`/`JSON.parse(...)['features']` do script GEE:
    recebe uma string com um GeoJSON (FeatureCollection, Feature ou geometria
    pura) colada pelo usuário e devolve uma lista de (id_talhao, geometria)
    já em WGS84.

    Se o GeoJSON tiver o membro legado "crs" (comum em exports do QGIS
    para projeções diferentes de WGS84, ex.: SIRGAS 2000/Brasil Polyconic
    - EPSG:5880), ele é respeitado e a geometria é reprojetada
    corretamente. Sem esse membro, assume-se o padrão GeoJSON (WGS84) —
    mas, como salvaguarda, um erro claro é levantado se as coordenadas
    obviamente não forem graus decimais (ex.: valores na casa dos
    milhões), em vez de gerar uma geometria corrompida silenciosamente.
    """
    dados = json.loads(texto_geojson)
    crs_epsg_explicito = _extrair_epsg_do_membro_crs(dados)

    if dados.get("type") == "FeatureCollection":
        gdf = gpd.GeoDataFrame.from_features(dados["features"])
    elif dados.get("type") == "Feature":
        gdf = gpd.GeoDataFrame.from_features([dados])
    else:
        # geometria "pura" (Polygon, MultiPolygon, ...), sem properties
        geom = shape(dados)
        gdf = gpd.GeoDataFrame({"geometry": [geom]})

    if crs_epsg_explicito:
        gdf = gdf.set_crs(crs_epsg_explicito, allow_override=True)
    else:
        coordenada_exemplo = _primeira_coordenada(gdf.geometry.iloc[0])
        if _parece_sistema_projetado(coordenada_exemplo):
            raise ValueError(
                f"O GeoJSON não possui o membro 'crs', mas as coordenadas "
                f"(ex.: {coordenada_exemplo}) estão na casa dos "
                "milhares/milhões — incompatível com WGS84 (graus "
                "decimais, sempre entre -180 e 180). Adicione o membro "
                '\'"crs": {"type": "name", "properties": {"name": '
                '"urn:ogc:def:crs:EPSG::XXXX"}}\' no GeoJSON com o EPSG '
                "correto (ex.: 5880 para SIRGAS 2000 / Brasil Polyconic), "
                "ou use `carregar_wkt_texto` informando o `crs_origem`."
            )
        gdf = gdf.set_crs(epsg=4326, allow_override=True)

    return _geometria_unica_a_partir_de_gdf(gdf, id_campo=id_campo)


def carregar_wkt_texto(texto_wkt, crs_origem, id_campo=None, propriedades=None):
    """
    Carrega uma geometria a partir de texto WKT (Well-Known Text) — ex.:
    "MULTIPOLYGON Z (((x y z, x y z, ...)))" — formato exportado por
    QGIS, PostGIS, ArcGIS, etc. Retorna lista de (id_talhao, geometria)
    em WGS84, no mesmo formato de `carregar_geojson_texto`.

    IMPORTANTE: diferente do GeoJSON e do KML, o WKT não guarda a
    projeção (CRS) junto do texto — por isso `crs_origem` é
    OBRIGATÓRIO. Informe o EPSG da projeção em que as coordenadas foram
    geradas, por exemplo:
      - "EPSG:4326"  -> já em graus decimais (WGS84)
      - "EPSG:5880"  -> SIRGAS 2000 / Brasil Polyconic (falso Easting
                        5.000.000, falso Northing 10.000.000 - é o caso
                        de coordenadas na casa dos milhões como
                        "5150073.16 6402741.07")
      - "EPSG:319XX" -> SIRGAS 2000 / UTM (XX = zona, ex. 31982 = zona 22S)

    Geometrias com Z (3D) são aceitas e convertidas para 2D
    automaticamente, já que o restante do fluxo (Treemap, busca STAC)
    trabalha apenas em X/Y.
    """
    if not crs_origem:
        raise ValueError(
            "O WKT não contém informação de projeção (CRS). Informe o "
            "parâmetro `crs_origem` com o EPSG correto — por exemplo "
            "'EPSG:5880' (SIRGAS 2000 / Brasil Polyconic) para "
            "coordenadas na casa dos milhões, ou 'EPSG:4326' se já "
            "estiver em graus decimais."
        )

    try:
        geometria = shapely_wkt.loads(texto_wkt)
    except Exception as e:
        raise ValueError(f"Não foi possível interpretar o texto como WKT válido: {e}")

    if geometria.has_z:
        geometria = shapely.force_2d(geometria)

    primeiro_anel = geometria.geoms[0].exterior if hasattr(geometria, "geoms") else geometria.exterior
    coordenada_exemplo = list(primeiro_anel.coords)[0]

    crs_normalizado = str(crs_origem).upper().replace(" ", "")
    if crs_normalizado in ("EPSG:4326", "4326") and _parece_sistema_projetado(coordenada_exemplo):
        raise ValueError(
            f"As coordenadas do WKT (ex.: {coordenada_exemplo}) estão na "
            "casa dos milhares/milhões — isso é incompatível com "
            "EPSG:4326 (graus decimais, sempre entre -180 e 180). Você "
            "está usando um sistema de coordenadas PROJETADO; informe o "
            "EPSG correto em `crs_origem` (ex.: 'EPSG:5880' para SIRGAS "
            "2000 / Brasil Polyconic, que usa exatamente essa faixa de "
            "valores, com falso Easting 5.000.000 e falso Northing "
            "10.000.000)."
        )

    propriedades = dict(propriedades) if propriedades else {}
    gdf = gpd.GeoDataFrame([propriedades], geometry=[geometria], crs=crs_origem)
    return _geometria_unica_a_partir_de_gdf(gdf, id_campo=id_campo)


def carregar_arquivo_geometria(caminho_arquivo, id_campo=None, crs_origem=None):
    """
    Carrega geometria(s) de talhão a partir de um arquivo .geojson, .kml
    ou .wkt no disco. Retorna lista de (id_talhao, geometria) em WGS84.

    `crs_origem` só é necessário (e obrigatório) para arquivos .wkt, já
    que esse formato não guarda a projeção junto do arquivo — veja
    `carregar_wkt_texto` para o significado do parâmetro.
    """
    caminho_arquivo = Path(caminho_arquivo)
    extensao = caminho_arquivo.suffix.lower()

    if extensao == ".wkt":
        texto = caminho_arquivo.read_text(encoding="utf-8").strip()
        return carregar_wkt_texto(texto, crs_origem=crs_origem, id_campo=id_campo)

    if extensao not in (".geojson", ".json", ".kml"):
        raise ValueError(
            f"Formato '{extensao}' não suportado. Use .geojson, .kml ou .wkt."
        )

    gdf = gpd.read_file(caminho_arquivo)
    if gdf.empty:
        raise ValueError(f"O arquivo '{caminho_arquivo.name}' não contém nenhuma feição.")

    return _geometria_unica_a_partir_de_gdf(gdf, id_campo=id_campo)


def buffer_metros(geometria_wgs84, distancia_m):
    """
    Aplica um buffer em METROS a uma geometria em WGS84, projetando
    temporariamente para uma projeção métrica local (Azimutal Equidistante
    centrada no centróide) — igual à técnica usada no gerador de covas da
    Treemap, garante precisão em qualquer ponto do Brasil.
    """
    centroide = geometria_wgs84.centroid
    crs_local = (
        f"+proj=aeqd +lat_0={centroide.y} +lon_0={centroide.x} "
        f"+x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    )
    serie = gpd.GeoSeries([geometria_wgs84], crs="EPSG:4326").to_crs(crs_local)
    serie_bufferizada = serie.buffer(distancia_m)
    return serie_bufferizada.to_crs(epsg=4326).iloc[0]