% Grupo 7 - Reconhecimento de Atividades Humanas
pkg load statistics
if(strcmp(computer(), "x86_64-pc-linux-gnu") )
  addpath("./libsvm/libsvm_x86_64-pc-linux-gnu");
elseif(ispc())
  addpath(".\libsvm\libsvm_windows");
endif

addpath("./ann");
addpath("./regression");

printf("Iniciando execucao.\n");
clear all, clc, close all;

% inicialmente ja fizemos pre-processamento dos dados para tentar diminuir a dimens�o de atributos e amostras
% retirando inconsistencias, redundancias, c�lulas nulas e fazendo a normalizacao de valores
% agora recuperamos os dados pre_processados que foram salvos no arquivo pre_processed

printf("\nCarregando dados pre-processados...\n");
fflush(stdout);


if(exist ("./data/pre_processed.mat.zip", "file") )
  printf("\nCarregando dados pre-processados...\n");
  fflush(stdout);
  load("./data/pre_processed.mat.zip");
else
  printf("Detectado que o pre-processamento ainda nao fora executado, aguarde, pre-processando....\n\n");
  preprocessing;
endif

if(exist ("./data/data_pca.mat.zip", "file") )
  load("./data/data_pca.mat.zip");
  printf("Dados do PCA Carregados !!!\n\n");
else
  # passando os dados pelo PCA para reduzir os atributos e utilizar em algoritmos que demandam mais processamento
  data_pca = pca(all_data);
  save("-zip", "./data/data_pca.mat.zip", "data_pca");
endif


% proveniente do 10-fold cross validation utilizado para a escolha de melhor fold
ksize = floor(rows(all_data)/10);

ktest = [];
ktrain = [];
ktest_pca = [];
ktrain_pca = [];

% knn com melhor k encontrado com grid search ----------------------------------------------------------
k = 1;
printf("\nIniciando execucao do knn com k = %d\n", k);
fflush(stdout);

tic;
for i = 1:rows(ktest)
  ypred_knn_test(i) = knn(ktrain(:,1:end-1), ktrain(:,end), ktest(i,1:end-1), k);
endfor
toc;
acc_knn = mean(double(ypred_knn_test == ktest(:,end))) * 100;
printf("\nAcuracia do knn para a base de teste: %.2f\n", acc_knn);
fflush(stdout);

printf("\nKNN finalizou execucao. Pressione enter para continuar...\n");
pause;

% regressao logistica com melhor lambda encontrado com grid search ---------------------------------------------------
lambda = 1;
printf("\nIniciando execucao de regressao logistica com lambda = %d\n", lambda);
fflush(stdout);
tic;
ypred_rl_test = regression(ktrain(:,1:end-1), ktrain(:,end), ktest(:,1:end-1), lambda);
toc;
acc_reg = mean(double(ypred_rl_test == ktest(:,end))) * 100;
printf('\nAcuracia da regressao para a base de teste: %.2f\n', acc_reg);
fflush(stdout);

printf("\nRegressao logistica finalizou execucao. Pressione enter para continuar...\n");
pause;

% redes neurais com o melhor lambda, max_iter encontrados com grid search -------------------------------------
lambda = 1;
max_iter = 1;
printf('\nIniciando execucao de redes neurais\n');

printf('\nO algoritmo redes neurais finalizou a execucao. Pressione enter para continuar...\n'); 
pause;

% svm com melhor c e gamma encontrados com grid search -----------------------------------------
c = 1;
gamma = 1;
printf('\nIniciando execucao de SVM com c = %.2f e g = %.2f\n', c, gamma);
fflush(stdout);
tic;
[ypred_svm_test, ~, ~] = svm(ktrain_pca(:,1:end-1), ktrain_pca(:,end), ktest_pca, c, g);
acc_svm = mean(double(y_pred == ktest(:,end))) * 100;
toc;
printf('\nAcuracia do svm para a base de teste: %.2f\n', acc_svm);
fflush(stdout);

printf('\nO algoritmo SVM finalizou a execucao. Pressione enter para continuar...\n'); 
pause;