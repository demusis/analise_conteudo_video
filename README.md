# Módulo para análise de Conteúdo de Vídeo

## 1\. Descrição Geral

O **Analisador de Conteúdo de Vídeo** é uma ferramenta de software, desenvolvida como uma aplicação web monolítica utilizando o microframework **Flask**, destinada à análise qualitativa de vídeos. A aplicação permite que um utilizador carregue um arquivo de vídeo, navegue por seu conteúdo com precisão (frame a frame), capture quadros (frames) de interesse, categorize-os, anote-os e aplique filtros de processamento de imagem em tempo real. Os dados gerados, incluindo os frames e um relatório de metadados, podem ser exportados para uso externo.

A ferramenta foi concebida para operar em sessões de análise discretas. O estado da aplicação (vídeo carregado, frames capturados e anotações) é mantido em memória, sendo reiniciado a cada novo upload de vídeo ou ao reiniciar o servidor. As categorias, no entanto, podem ser persistidas em um arquivo `JSON`.

-----

## 2\. Funcionalidades

  - **Upload e Visualização:** Carregamento de um arquivo de vídeo local para análise em um player interativo.
  - **Navegação Precisa:** Controles para reprodução, pausa, ajuste de velocidade e navegação frame a frame.
  - **Captura de Frames:** Extração de quadros específicos com base no `timestamp` do vídeo.
  - **Gestão de Categorias:** Criação, edição, exclusão, importação e exportação de categorias para classificar os frames capturados.
  - **Anotação:** Associação de notas textuais a cada frame capturado.
  - **Galeria de Frames:** Visualização de todos os frames capturados em uma galeria filtrável por categoria.
  - **Processamento de Imagem:** Aplicação não destrutiva e em tempo real de uma sequência de filtros de imagem nos frames capturados:
      - Ajuste de Brilho e Contraste.
      - Equalização de Histograma Adaptativa por Contraste Limitado (CLAHE).
      - Balanço de Branco automático.
  - **Análise de Metadados:** Exibição de informações detalhadas do arquivo de mídia (utilizando `MediaInfo`) e cálculo do hash SHA-512 do arquivo.
  - **Exportação de Dados:**
      - **ZIP:** Exporta os frames capturados (com filtros aplicados) em um arquivo `.zip`, organizados em subpastas correspondentes às suas categorias.
      - **CSV:** Exporta um relatório contendo o `timestamp`, categoria, nome do arquivo e anotações de cada frame.
  - **Persistência de Anotações:** Capacidade de exportar e importar o conjunto de anotações (galeria) de uma sessão, permitindo a continuidade do trabalho.

-----

## 3\. Tecnologias Utilizadas

### Backend

  - **Python 3**
  - **Flask:** Microframework web para o servidor e rotas da API.
  - **PyAV:** Biblioteca para decodificação e extração precisa de frames de vídeo.
  - **OpenCV-Python (`cv2`):** Biblioteca para as operações de processamento de imagem.
  - **NumPy:** Suporte para manipulação de arrays multidimensionais nas rotinas de imagem.
  - **Pillow (PIL):** Utilizada para a manipulação básica de imagens.
  - **Pandas:** Empregado para a estruturação e exportação dos dados para o formato `.csv`.
  - **PyMediaInfo:** Wrapper para a ferramenta `MediaInfo`, usada para extrair metadados detalhados dos arquivos de vídeo.

### Frontend

  - **HTML5 / CSS3 / JavaScript (ES6+):** Estrutura, estilo e lógica da interface do utilizador.
  - **Tailwind CSS:** Framework CSS para a estilização rápida da interface.
  - **SortableJS:** Biblioteca para habilitar a reordenação de elementos (filtros de imagem) via arrastar e soltar.

-----

## 4\. Instalação e Execução

Para executar a aplicação localmente, siga os passos abaixo.

### Pré-requisitos

  - Python 3.8 ou superior.
  - `pip` (gerenciador de pacotes do Python).
  - A ferramenta `MediaInfo` deve estar instalada no sistema e acessível no `PATH` do sistema para que a funcionalidade de exibição de metadados opere corretamente.



### Guia Rápido de Instalação e Uso

> Estes passos funcionam para Windows, macOS e distribuições Linux.
> **Python ≥ 3.9** é recomendado.

---

#### 1. Primeira Instalação (setup inicial)

| Etapa                              | Comando                                                                                                             | Observações                                                                                                            |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **1.1 Clonar ou baixar o projeto** | `bash<br># com Git<br>git clone https://github.com/seu-usuario/seu-repo.git<br># ou baixe o zip e extraia<br>`      | Substitua a URL pelo seu repositório real.                                                                             |
| **1.2 Criar o ambiente virtual**   | `bash<br>python -m venv venv<br>`                                                                                   | O diretório `venv/` será criado na raiz do projeto.                                                                    |
| **1.3 Ativar o ambiente virtual**  | **Windows**  <br>`bash<br>.\venv\Scripts\activate<br>`  **macOS / Linux**<br>`bash<br>source venv/bin/activate<br>` | Você saberá que o ambiente está ativo quando aparecer `(venv)` antes do prompt.                                        |
| **1.4 Instalar dependências**      | `bash<br>pip install -r requirements.txt<br>`                                                                       | O arquivo `requirements.txt` já contém:<br>`Flask`, `av`, `Pillow`, `pandas`, `pymediainfo`, `opencv-python`, `numpy`. |
| **1.5 Executar a aplicação**       | `bash<br>python app.py  # ou<br>flask run --debug<br>`                                                              | O modo **debug** recarrega o servidor a cada alteração.                                                                |

Abra o navegador em **[http://127.0.0.1:5000](http://127.0.0.1:5000)** para ver a interface.

---

#### 2. Sessões Futuras (após o primeiro dia)

Sempre que quiser usar o projeto novamente:

1. Abra um terminal/cmd dentro da pasta do projeto.
2. **Ative** o ambiente virtual:

   ```bash
   # Windows
   .\venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```
3. Rode o app:

   ```bash
   python app.py
   ```
4. Quando terminar, **desative** o ambiente:

   ```bash
   deactivate
   ```

---

#### 3. Dicas Extras

* **Porta ou host diferentes**

  ```bash
  flask run --host 0.0.0.0 --port 8000
  ```
* **Variáveis de ambiente do Flask** (opcional):

  ```bash
  # Unix-like
  export FLASK_APP=app.py
  export FLASK_ENV=development
  # Windows (PowerShell)
  set FLASK_APP=app.py
  set FLASK_ENV=development
  ```
* **Atualizar dependências**

  ```bash
  pip install --upgrade -r requirements.txt
  ```
* **Excluir o ambiente virtual**
  Basta apagar a pasta `venv/` e recriá-la quando necessário.

Com isso, você terá um README mais completo, cobrindo tanto a configuração inicial quanto o fluxo de trabalho diário.



-----

## 5\. Estrutura do Projeto e Funcionamento

O projeto consiste em um único arquivo `app.py` que contém tanto a lógica do backend (Flask) quanto o código do frontend (HTML, CSS, JavaScript) embutido.

  - `app.py`: Arquivo principal que define as rotas da API, a lógica de negócio e renderiza a interface.
  - `/data`: Diretório criado automaticamente na primeira execução.
      - `/data/videos/`: Armazena temporariamente o vídeo carregado pelo utilizador.
      - `/data/frames/`: Armazena os frames capturados como arquivos de imagem (`.png`).
      - `/data/categories.json`: Arquivo que armazena as categorias criadas pelo utilizador.

### Fluxo de Trabalho Típico

1.  O utilizador seleciona um arquivo de vídeo através da interface.
2.  O vídeo é enviado para o servidor, que o armazena em `/data/videos/` e limpa os dados da sessão anterior.
3.  O vídeo é carregado no player da interface.
4.  No painel direito, o utilizador pode criar categorias para a análise.
5.  Durante a visualização, o utilizador pausa no momento de interesse e seleciona uma categoria para a captura.
6.  Ao clicar em "Capturar Frame", o servidor extrai o quadro exato, salva-o em `/data/frames/` e adiciona seus metadados à sessão.
7.  O frame capturado aparece na galeria na parte inferior, onde é possível adicionar uma nota ou alterar sua categoria.
8.  Ao clicar em "Abrir" em um frame da galeria, um modal é exibido, permitindo a aplicação e ajuste de filtros de imagem. As alterações são salvas automaticamente.
9.  Ao final da análise, o utilizador pode exportar um arquivo `.zip` com as imagens ou um `.csv` com o relatório.

-----

## 6\. Limitações

  - **Armazenamento em Memória:** Os dados da sessão de análise (frames capturados, anotações) são voláteis. Reiniciar o servidor ou carregar um novo vídeo resultará na perda dos dados da sessão corrente. Apenas as categorias são salvas em disco.
  - **Sessão Única:** A aplicação foi projetada para analisar um vídeo por vez. Não há suporte para múltiplas sessões ou utilizadores simultâneos.
  - **Processamento de Arquivos Grandes:** O upload e processamento de arquivos de vídeo muito grandes podem consumir uma quantidade elevada de recursos do sistema (CPU e RAM).
