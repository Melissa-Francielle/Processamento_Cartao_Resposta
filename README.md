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
Esta etapa tem como objetivo preparar as imagens adquiridas para as fases subsenquentes melhorando sua qualidade e estrutura para facilitar a análise.as imagens são convertidas para escalas de cinza e, passa pela binarização de Otsu, utilizando o parâmetro
`cv2.THRESH_BINARY_INV + Otsu`. Além de ser feito uma melhoria nos elementos gráficos da imagem, também é identificado os vértices dos triângulos utilizando a função `_localizar_triangulos()`.

- Segmentação:
Na fase de segmentação, as imagens pré-processadas são analisadas com o objetivo de isolar as regiões relevantes, neste caso, as linhas contendo apenas as questões indiivduais. O cartão de resposta é dividido em três colunas verticais durante o processamento.Em cada coluna, os contornos com formato aproximado de quadrados são identificados por meio da função `_square_contours`. Após a identificação das linhas, cada linha de questão é recortada e salva como um arquivo de imagem.

- Interpretação:
A fase de intepretação é responsável pela analse dos recortes obtidos na etapa da segmentação e extrair a partir disso as alternativas assinaladas em cada questão. As operações realizadas nessa fase correspondem ao módulo de reconhecimento óptico de marcações.Cada questão individual calcula-se o preenchimento no interior (miolo) dos cincos quadrados correspondetes às alternativas. A função `extrair_id_pasta` realiza a extração desse número, possibilitando que as respostas sejam indexadas corretamente.
\item \textbf{Comparação com Gabarito:} Por fim, as respostas extraidas são comparadas com o gabarito oficial, que está armazenado em um arquivo CSV

1. [Etapa 1 - OMR: Corte, Filtros e Identificação de Triângulos](https://github.com/Melissa-Francielle/Processamento_Cartao_Resposta/blob/main/OMR.py)

2. [Etapa 2 - Verificação das Respostas](https://github.com/Melissa-Francielle/Processamento_Cartao_Resposta/blob/main/verificar_respostas.py)

3. [Etapa 3 - Comparação e Análise das Métricas](https://github.com/Melissa-Francielle/Processamento_Cartao_Resposta/blob/main/Artigo_Processamento_de_Cartão_Resposta.pdf)

## Artigo 
Para melhor compreensão da implementação do sistema de processamento de cartões de resposta, recomenda-se a leitura do seguinte artigo:
[Artigo - Processamento de Cartão Resposta Utilizando Python](https://github.com/Melissa-Francielle/Processamento_Cartao_Resposta/blob/main/Artigo_Processamento_de_Cartão_Resposta.pdf)
