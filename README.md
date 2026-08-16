# 🛰️ Buscador de Imagens Anuais

Dashboard para visualizar o histórico de imagens de satélite de um talhão, ano a ano, lado a lado — sem precisar de banco de dados.

## O que faz

- Recebe a geometria de um talhão colada como **GeoJSON** ou **WKT**, ou enviada como arquivo **.geojson / .kml / .wkt**.
- Busca automaticamente, para cada ano, a imagem **menos nublada** disponível no [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) (STAC) — Landsat para os anos mais antigos (desde 1984), Sentinel-2 para os anos recentes (mais detalhe).
- Mostra tudo em uma **grade** (3 colunas por padrão), em falsa cor **NIR/SWIR1/RED**, com o contorno do talhão sobreposto.
- Sugere automaticamente **anos candidatos a colheita** (via queda de NDVI ano a ano), destacando a borda desses anos em vermelho na grade.
- Acesso ao Planetary Computer é **público e gratuito** — não precisa de conta, login nem chave de API.

## Como rodar localmente

```bash
pip install -r requirements.txt
streamlit run dashboard_imagens.py
```

Abre em `http://localhost:8501`.

## Deploy online (gratuito)

Este projeto pode ser hospedado de graça no [Streamlit Community Cloud](https://share.streamlit.io):

1. Suba este repositório no GitHub (você já fez isso 🎉).
2. Acesse [share.streamlit.io](https://share.streamlit.io) e conecte sua conta do GitHub.
3. Clique em **"New app"**, escolha este repositório e o arquivo principal `dashboard_imagens.py`.
4. Deploy — você recebe uma URL pública para compartilhar.

## Arquivos do projeto

| Arquivo | O que faz |
|---|---|
| `dashboard_imagens.py` | Interface do dashboard (Streamlit) |
| `composicao_anual.py` | Busca e processamento das imagens de satélite via STAC |
| `geometria_utils.py` | Leitura e validação de geometrias (GeoJSON, WKT, KML) |
| `deteccao_colheita.py` | Sugestão automática de ano de colheita via NDVI |
| `requirements.txt` | Dependências do projeto |

## Fonte de dados

[Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) — catálogo STAC público, coleções `landsat-c2-l2` (Landsat 4/5/7/8/9, 1984-hoje) e `sentinel-2-l2a` (Sentinel-2, 2015-hoje).

## Observações

- Não há banco de dados — o único "cache" é um diretório local de arquivos `.pkl`, usado só para não repetir buscas já feitas na mesma sessão.
- Em hospedagem gratuita (Streamlit Community Cloud), esse cache local é apagado sempre que o app reinicia — é o esperado, já que não há disco persistente no plano gratuito.
