# NASMSimBeam - Simulação de Feixes Ópticos para Comunicação FSO

Este projeto tem como objetivo modelar, simular e analisar a propagação de feixes ópticos, especialmente feixes de base Gaussianas, em sistemas de comunicação por espaço livre (Free Space Optics - FSO) Homogêneos. A simulação considera parâmetros iniciais como cintura inicial e distância de propagação, e demonstra a variação da difração, da intensidade, da fase e a variação provocada por fendas no trajeto do feixe.

## 📁 Estrutura do Projeto

├── DFT-propag/

│ ├── feixe_dft_fase_HG.py

│ ├── feixe_dft_fase.py

│ ├── feixe_dft_intensidade_HG.py

│ └── feixe_dft_intensidade.py

├── DFT-valid_fendas/

│ ├── fendacir.py

│ ├── fendaret.py

│ └── fendartri.py

├── DFT-valid_graficos_wz/

│ └── feixigaussiano.py

├── graficos.py

├── main.py

├── taxaerro.py

└── README.md


## 📌 Descrição dos Principais Módulos

### `DFT-propag/`
Contém scripts para simulação da propagação do feixe via Transformada Discreta de Fourier (DFT), tanto em fase quanto em intensidade. Também inclui versões para feixes Hermite-Gaussianos (HG).

### `DFT-valid_fendas/`
Scripts para simular diferentes tipos de obstruções (fendas) no caminho do feixe:
- `fendacir.py`: Fenda circular
- `fendaret.py`: Fenda retangular
- `fendartri.py`: Fenda triangular

### `DFT-valid_graficos_wz/`
Contém `feixigaussiano.py`, que realiza os cálculos dos resultados da propagação dos feixes com o método NASM e com o método analítico ao longo da distância z.

### `graficos.py`
Geração de gráficos das cinturas finais em relação a distância de propagação dos feixes obtidos pelo método NASM e analitico.

### `taxaerro.py`
Geração de graficos que compara os valores das cinturas finais em relação a propagação obtidas com o método NASM e com o metodo análitico.

### `main.py`
Arquivo principal de execução. Realiza a chamada dos métodos de propagação e visualização gráfica.

## ⚙️ Requisitos

- Python 3.12+
- Bibliotecas:
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `os`, `sys`

Instale as dependências com:

```bash
pip install -r requirements.txt
 ```

## 🚀 Como Executar
1. Clone este repositório:

``` bash 
git clone https://github.com/seu_usuario/NASMSimBeam.git
```

2. Navegue até o diretório:
```bash
cd NASMSimBeam
```

3. Execute o script principal:

``` bash
python main.py
```

## 📊 Resultados Esperados
* Visualizações da intensidade e fase antes e apos a propagação do feixe.
* Vizualização dos graficos com os valores das cinturas finais, de cada método e da comparação entre eles
* Vizualização dos feixes exatamente antes de entrar nas fendas e apos atravessar as fendas em relação a distância de propagação (z)

## 📚 Referências
* GOODMAN, Joseph W. Introduction to Fourier optics. Roberts and Company publishers, 2005.

