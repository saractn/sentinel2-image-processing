# Prestação de serviços de gestão e organização de dados para execução de código de processamento de imagens Sentinel-2 para deteção de alterações.

## Referência
Ref.ª ADG-29-2026

## Base Legal
Ajuste direto ao abrigo do Art.º 20.º, n.º 1, alínea d) do Código dos Contratos Públicos (CCP)

---

## Descrição do Projeto

Este repositório contém os scripts Python desenvolvidos para o pré-processamento, organização e conversão de dados geoespaciais provenientes de imagens Sentinel-2, com vista à deteção de alterações no território. Os serviços prestados visam garantir a interoperabilidade dos dados e o processamento eficiente de grandes volumes de informação, suportando a criação de bases de dados territoriais e análises espaciais.

---

## Serviços Prestados

### Tarefa 1 – Pré-processamento e descarregamento de imagens
- Seleção de imagens da coleção Sentinel-2 SR (Surface Reflectance) para um determinado intervalo de datas.  
- Utilização do produto **S2_CLOUD_PROBABILITY (s2cloudless)** para mascarar áreas afetadas por nuvens e sombras.  
- Transferência das imagens em formato **GeoTIFF**.  
- Garantia de que os valores válidos de refletância são inteiros entre 0 e 10000, sendo os valores **NODATA** convertidos para 65535.

### Tarefa 2 – Conversão para HDF5
- Utilização como inputs de ficheiros GeoTIFF correspondentes a uma tile e a uma data de aquisição, bem como de um ficheiro vetorial de delimitação de Portugal Continental.  
- Geração como output de um ficheiro **HDF5** que representa a série temporal dos dados GeoTIFF com extensão espacial correspondente ao input vetorial.  
- Representação de valores **NODATA** como 65535.

### Tarefa 3 – Conversão para Parquet
- Conversão de ficheiros em formato **Parquet**, em que cada linha representa um pixel e uma data, com atributos adicionais.  
- Garantia de que os pixels não representados nos ficheiros Parquet incluem as respetivas coordenadas e valores NODATA para os restantes atributos.  

---
