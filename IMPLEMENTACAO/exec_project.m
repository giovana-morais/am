% Grupo 7 - Reconhecimento de Atividades Humanas

printf("Iniciando execucao.\n");
clear all, clc, close all;
% inicialmente ja fizemos pre-processamento dos dados para tentar diminuir a dimensï¿½o de atributos e amostras
% retirando inconsistencias, redundancias, cï¿½lulas nulas e fazendo a normalizacao de valores
% agora recuperamos os dados pre_processados que foram salvos no arquivo pre_processed

printf("\nCarregando dados pre-processados...\n");
fflush(stdout);
load("pre_processed.mat");

% em seguida, devemos dividir os dados gerais para validacao cruzada, com k-fold sendo 10 (mais usualmente utilizado em AM)
% assim sendo, 9 partes irao para treinamento enquanto apenas 1 ira para teste (isso ocorre 10 vezes, para cada algoritmo)
% aqui conseguiremos avaliar as hipoeses e selecionar a melhor

ksize = floor(length(all_data)/10);

printf("\nIniciando grid do 10-fold cross-validation\n");

for iter = 1:9
  if iter ~= 1 && iter ~= 9
    ktest = all_data(iter*ksize:((iter*ksize)+ksize), :);
    ktrain = [all_data(1:((ksize*iter)-1),:); all_data(((iter*ksize)+ksize+1):end, :)];
  elseif iter == 1
    ktest = all_data(1:ksize, :);
    ktrain = all_data((ksize+1):end, :);
  elseif iter == 9
    ktest = all_data(iter*ksize:end, :);
    ktrain = all_data(1:((ksize*iter)-1), :);
  endif
  
  printf("\nIteracao com ktest = %d, o tamanho de ktest eh %d e o tamanho de ktrain eh %d\n", iter, length(ktest), length(ktrain));

  % execucao do knn
  printf('\nIniciando execucao do knn\n');
  fflush(stdout);

  % escolhemos o k com maior acuracia
  % aqui utilizamos apenas numeros impares para nao ter que haver desempate nos k neighbours mais proximos

  % spoiler: maior acuracia acontece pra k = 1 (ja testei com 200, com 300, com 400, com 900 e só vai diminuindo a acuracia)

  for k = 1:2:50
    j = 1;
    printf("\nPara k = %d\n", k);
    fflush(stdout);
    ac = 0;
    tic();
    for i = 1:length(ktest)
      %printf("nao bugou %d \n", i);
      %fflush(stdout);
      ypred = knn(ktrain(:,1:end-1), ktrain(:,end), ktest(i,1:end-1), k);
      if(ypred == ktest(i, end))
        ac += 1;
      endif
    endfor
    toc();
    acuracyknn(j) = ac/length(ktest);
    printf("Ocorre %.2f%% de acuracia\n", acuracyknn(j)*100);
    fflush(stdout);
    j += 1;
    % TODO: plotar o mapa da distribuicao y pro relatorio
    % pq essa distribuicao aparentemente eh temporal
  endfor

  printf("\nO k que apresentou maior acuracia foi: %d\n\n", find(acuracyknn == max(acuracyknn)));


  fprintf('\nO algoritmo KNN finalizou a execucao. Pressione enter para continuar.\n');
  pause;

  % execucao da regressao logistica
  printf('\nIniciando execucao da regressao logistica\n');
  fflush(stdout);

  %printf("logistic regression\n");
  %fflush(stdout);
  %regression(ktrain(:,1:end-1), ktrain(:,end));

  fprintf('\nO algoritmo de Regressao Logistica finalizou a execucao. Pressione enter para continuar.\n');
  pause;

  % execucao da regressao logistica
  printf('\nIniciando execucao de redes neurais artificiais\n');
  fflush(stdout);

  fprintf('\nO algoritmo de Redes Neurais Artificiais finalizou a execucao. Pressione enter para continuar.\n');
  pause;

  % execucao da regressao logistica
  printf('\nIniciando execucao de SVM\n');
  fflush(stdout);

  fprintf('\nO algoritmo SVM finalizou a execucao. Pressione enter para continuar.\n');
  pause;
  
endfor