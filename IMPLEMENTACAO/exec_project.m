% Grupo 7 - Reconhecimento de Atividades Humanas

printf("Iniciando execucao.\n");
% inicialmente ja fizemos pre-processamento dos dados para tentar diminuir a dimensão de atributos e amostras
% retirando inconsistencias, redundancias, células nulas e fazendo a normalizacao de valores
% agora recuperamos os dados pre_processados que foram salvos no arquivo pre_processed

printf("\nCarregando dados pre-processados...\n");
fflush(stdout);
load("pre_processed.mat");

% em seguida, devemos dividir os dados gerais para validacão cruzada, com k sendo 10 (mais usualmente utilizado em AM)
% assim sendo, 9 partes irao para treinamento enquanto apenas 1 ira para teste (isso ocorre 10 vezes, para cada algoritmo)
% aqui conseguiremos avaliar as hipóteses e selecionar a melhor 

