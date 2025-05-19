# UNIVERSIDADE ESTADUAL DO NORTE DO PARANÁ - UENP 
## Trabalho do 4º ano da matéria de Computação Gráfica do Curso de Ciência da Computação 

Docente: Wellington Della Mura

Discentes: Joana Shizu e Melissa Francielle

## Estrutura do trabalho

* ### Problema
A partir das imagens de amostragem (498 arquivos) faça um algoritmo de visão computacional capaz de isolar a área do cartão resposta com as marcações (60 questões) e interprete cada marcação (5 alternativas).
Use o arquivo de respostas para medir a taxa de acertos do seu algoritmo.

   ![Imagem Ilustrativa](https://d23vy2bv3rsfba.cloudfront.net/listas/1_f5eaa519f168b122b06ae02e55401bee_5804.png)
                                   (Imagem Ilustrativa)
* ### Hipótese
O problema de análise dos cartões de resposta pode ser resolvido utilizando um algoritmo de visão computacional desenvolvido em linguagem Python, sendo capaz de isolar as questões e interpretar as alternativas de cada cartão resposta. O algoritmo será testado por meio da comparação dos resultados obtidos com o arquivo resposta fornecido, permitindo calcular a taxa de acerto. 

* ### Método
Utilizando um protocolo para padronizar o processamento das imagens de forma que todas as imagens sejam tratadas da mesma forma e garanta o funcionamento do programa para identificar alternativas marcadas no cartão respostas pelos candaditos.
  * ### Protocolo:
- Aquisição: Os quatrocentos e noventa e nove cartões de resposta foram digitalizados utilizando scanner com resolução de 300 DPI, snedo aplicado a binarização automatica das imagens, resultando em arquivos JPEG preto e branco prontos para o processamneto. Cada imagem está associada a um registro no arquivo CSV contendo o gabarito de respostas, utilizado para validar a extração das próximas etapas.
  
- Pré-processamento:
Padronização de escala: as imagens de cada cartão é redimensionada para um tamanho 800 x 800, utilizando cv2.resize, no intuito de normalizar as proporções para visualização. 
Conversão de tons de cinza: utilizando o seguinte comando `cv2.cvtColor(...,COLOR_BGR2GRAY)` tornando o processamento mais simples e mais rápido. 
Binarização: além da alteração da coloração é feito uma binarização dos cartões utilizando um comando `cv2.threshold(..., THRESH_BINARY_INV)` isolando as regiões mais escuras como o caso dos triângulos criando uma máscara binária apropriada para os contornos.

- Segmentação:
Detecção de contornos externos: utilizando o `cv2.findContours` para localização dos objetos presentes na imagem cortada e binarizada. 
Classificação dos quatros marcadores: é feito uma verificação de cada triângulo e a região que eles se encontram a partir da margem para rotular onde se encontram os “cantos” de cada triângulo para a próxima parte do processamento.
Retificação: assim que é encontrado os cantos que apresentam os triângulos do cartão resposta é feito então um reorder, que é calcular uma matriz transformando a perspectiva deixando somente visível a parte que lhe é interessante ao trabalho em questão.
Divisão das colunas: Assim que feito a retificação da imagem é então dividido em 3 colunas de cada cartão dos candidatos no intuito de tornar mais fácil ao algoritmo de encontrar as questões para somente após essa etapa realizar a análise e depois a verificação da taxa de acerto dos cartões. 
Divisão das linhas: após as colunas de cada cartao de cada candidato ser extraída e armazenada, é feito então a divisão das linhas, todas as 60 questões, que estão divididas nas 3 colunas cortadas do cartão. 

- Interpretação:
  Para cada grupo calcula-se o **Pixels Brancos**. A bolha (os pixels brancos) com uma razão de ≥ 0,30 e diferença ≥ 0,15 é considerada marcada. Aplicando o Tesseract utilizando a biblioteca  `pytesseract ` para ler as alternativas impressas sendo (A-E), esses resultados são gravados em um arquivo de formato csv dado como  `respostas_candidatos.csv ` no formato  `questão;resposta;candidato `. Os dados foram de suma importância para analisar que dependendo de certas interpretações do algoritmo resultava em outras respostas devido fatores de posicionamento, como os quadrados eram preenchidos, se havia alguma dificuldade de identificar o quadrado.

1. [Etapa 1 - OMR: Corte, Filtros e Identificação de Triângulos](https://github.com/Melissa-Francielle/Processamento_Cartao_Resposta/blob/main/OMR.py)

2. [Etapa 2 - Verificação das Respostas](https://github.com/Melissa-Francielle/Processamento_Cartao_Resposta/blob/main/verificar_respostas.py)

3. [Etapa 3 - Comparação e Análise das Métricas](https://github.com/Melissa-Francielle/Processamento_Cartao_Resposta/blob/main/comparar_respostas.py)


