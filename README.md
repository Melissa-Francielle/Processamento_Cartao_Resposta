# Estrutura

Problema

A partir das imagens de amostragem (498 arquivos) faça um algoritmo de visão computacional capaz de isolar a área do cartão resposta com as marcações (60 questões) e interprete cada marcação (5 alternativas).
Use o arquivo de respostas para medir a taxa de acertos do seu algoritmo.

Hipótese
O problema de análise dos cartões de resposta pode ser resolvido utilizando um algoritmo de visão computacional desenvolvido em linguagem Python, sendo capaz de isolar as questões e interpretar as alternativas de cada cartão resposta. O algoritmo será testado por meio da comparação dos resultados obtidos com o arquivo resposta fornecido, permitindo calcular a taxa de acerto. 

Método
Protocolo:
- Aquisição:
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

Experimento
- Conjunto de dados
- Aplicação

* Importantes
  - Taxa de sucesso
  - Tempo
  - Memória

# Conclusão

