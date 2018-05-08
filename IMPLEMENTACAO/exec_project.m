% Grupo 7 - Reconhecimento de Atividades Humanas

printf("Iniciando execucao.\n");
clear all, clc, close all;
% inicialmente ja fizemos pre-processamento dos dados para tentar diminuir a dimensão de atributos e amostras
% retirando inconsistencias, redundancias, células nulas e fazendo a normalizacao de valores
% agora recuperamos os dados pre_processados que foram salvos no arquivo pre_processed

printf("\nCarregando dados pre-processados...\n");
fflush(stdout);
load("pre_processed.mat");

% em seguida, devemos dividir os dados gerais para validacão cruzada, com k sendo 10 (mais usualmente utilizado em AM)
% assim sendo, 9 partes irao para treinamento enquanto apenas 1 ira para teste (isso ocorre 10 vezes, para cada algoritmo)
% aqui conseguiremos avaliar as hipóteses e selecionar a melhor 

ksize = floor(length(all_data)/10);
ktrain = all_data(1:ksize*9, :);
ktest = all_data(ksize*9:end, :);

printf('\nIniciando execucao do knn\n');
fflush(stdout);

% para escolher o k utilizamos o elbow method com k indo de 1 a 20
% consequentemente escolhemos o k minimo
% isso aqui vai demorar absurdos, to sem coragem no momento
for k = 1:20
  printf("\nPara k = %d\n", k);
  fflush(stdout);
  for i = 1:length(ktest)
    ypred = knn(ktrain(:,1:end-1), ktrain(:,end), ktest(i,1:end-1), k);
    if(ypred != ktest(i, end))
      error(i) = 1;
    endif
  endfor
  errorknn(k) = sum(error);
  printf("Ocorrem %d erros\n", errorknn(k));
  fflush(stdout);
endfor


