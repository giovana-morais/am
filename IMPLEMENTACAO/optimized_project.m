% Grupo 7 - Reconhecimento de Atividades Humanas
pkg load statistics

if(strcmp(computer(), "x86_64-pc-linux-gnu") )
  addpath("./libsvm/libsvm_x86_64-pc-linux-gnu");
elseif(ispc())
  addpath("./libsvm/libsvm-windows");
endif

addpath("./ann");
addpath("./regression");

printf("Iniciando execucao.\n");
clear all, clc, close all;

% inicialmente ja fizemos pre-processamento dos dados para tentar diminuir a dimens�o de atributos e amostras
% retirando inconsistencias, redundancias, c�lulas nulas e fazendo a normalizacao de valores
% agora recuperamos os dados pre_processados que foram salvos no arquivo pre_processed

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
  printf("Dados com PCA Carregados!\n\n");
else
  # passando os dados pelo PCA para reduzir os atributos e utilizar em algoritmos que demandam mais processamento
  data_pca = pca(all_data);
  save("-zip", "./data/data_pca.mat.zip", "data_pca");
endif


% proveniente do 10-fold cross validation utilizado para a escolha de melhor fold
ksize = floor(rows(all_data)/10);

iter = 4;

ktest = all_data((((iter-1)*ksize)+1):(((iter-1)*ksize)+ksize), :);
ktrain = [all_data(1:(ksize*(iter-1)),:); all_data((((iter-1)*ksize)+ksize+1):end, :)];
ktest_pca = data_pca((((iter-1)*ksize)+1):(((iter-1)*ksize)+ksize), :);
ktrain_pca = [data_pca(1:(ksize*(iter-1)),:); data_pca(((iter-1)*ksize+ksize+1):end, :)];

% linhas sao os algoritmos e colunas sao as metricas: f-medida, precisao, revocacao, acuracia, tempo
mat_res = zeros(5, 5);

% knn com melhor k encontrado com grid search ----------------------------------------------------------
k = 1;
printf("\nIniciando execucao do knn com k = %d\n", k);
fflush(stdout);

tic;
for i = 1:rows(ktest)
  ypred_knn_test(i, 1) = knn(ktrain(:,1:end-1), ktrain(:,end), ktest(i,1:end-1), k);
endfor
time_exec = toc;

[fknn, precknn, revknn] = fmeasure(ypred_knn_test, ktest(:, end));
acc_knn = mean(double(ypred_knn_test == ktest(:,end))) * 100;
printf("\nF-medida do knn para a base de teste: %.2f\n", fknn);
fflush(stdout);

mat_res(1,1) = fknn;
mat_res(1,2) = precknn*100;
mat_res(1,3) = revknn*100;
mat_res(1,4) = acc_knn;
mat_res(1,5) = time_exec;

printf("\nKNN finalizou execucao. Pressione enter para continuar...\n");
pause;

% regressao logistica com melhor lambda encontrado com grid search ---------------------------------------------------
lambda = 2;
printf("\nIniciando execucao de regressao logistica com lambda = %d\n", lambda);
fflush(stdout);
tic;
ypred_rl_test = regression(ktrain(:,1:end-1), ktrain(:,end), ktest(:,1:end-1), lambda);
time_exec = toc;

[freg, precreg, revreg] = fmeasure(ypred_rl_test, ktest(:, end));
acc_reg = mean(double(ypred_rl_test == ktest(:,end))) * 100;
printf("\nF-medida da regressao para a base de teste: %.2f\n", freg);
fflush(stdout);

mat_res(2,1) = freg;
mat_res(2,2) = precreg*100;
mat_res(2,3) = revreg*100;
mat_res(2,4) = acc_reg;
mat_res(2,5) = time_exec;

printf("\nRegressao logistica finalizou execucao. Pressione enter para continuar...\n");
pause;

% redes neurais com o melhor lambda, max_iter e hidden_neurons encontrados com grid search -------------------------------------
printf('\nIniciando execucao de redes neurais com 1 camada\n');
hidden_neurons_rn1 = 151;
max_iter_rn1 = 750;
lambda_rn1 = 1;

tic;
ypred_rn1_test = neural_network_1l(hidden_neurons_rn1, max_iter_rn1, ktrain_pca, ktest_pca, lambda_rn1);
time_exec = toc;

[frn1, precrn1, revrn1] = fmeasure(ypred_rn1_test, ktest_pca(:, end));
acc_rn1 = mean(double(ypred_rn1_test == ktest_pca(:,end))) * 100;
printf("\nF-medida da regressao para a base de teste: %.2f\n", frn1);
fflush(stdout);

mat_res(1,1) = frn1;
mat_res(1,2) = precrn1*100;
mat_res(1,3) = revrn1*100;
mat_res(1,4) = acc_rn1;
mat_res(1,5) = time_exec;

printf('\nO algoritmo redes neurais com 1 camada finalizou a execucao. Pressione enter para continuar...\n'); 
pause;

% redes neurais com o melhor lambda, max_iter e hidden_neuros encontrados com grid search -------------------------------------
printf('\nIniciando execucao de redes neurais com 2 camadas\n');
hidden_neurons_rn2 = 300;
max_iter_rn2 = 1000;
lambda_rn2 = 2;

tic;
ypred_rn2_test = neural_network_2l(hidden_neurons_rn2, max_iter_rn2, ktrain_pca, ktest_pca, lambda_rn2);
time_exec = toc;

[frn2, precrn2, revrn2] = fmeasure(ypred_rn2_test, ktest_pca(:, end));
acc_rn2 = mean(double(ypred_rn2_test == ktest_pca(:,end))) * 100;
printf("\nF-medida da regressao para a base de teste: %.2f\n", frn2);
fflush(stdout);

mat_res(1,1) = frn2;
mat_res(1,2) = precrn2*100;
mat_res(1,3) = revrn2*100;
mat_res(1,4) = acc_rn2;
mat_res(1,5) = time_exec;

printf('\nO algoritmo redes neurais com 2 camadas finalizou a execucao. Pressione enter para continuar...\n'); 
pause;


% svm com melhor c e gamma encontrados com grid search -----------------------------------------
c = 32;
g = 0.0078125;
printf('\nIniciando execucao de SVM com c = %.2f e g = %f\n', c, g);
fflush(stdout);
tic;
[ypred_svm_test, ~, ~] = svm(ktrain_pca(:,1:end-1), ktrain_pca(:,end), ktest_pca, c, g);
time_exec = toc;

[fsvm, precsvm, revsvm] = fmeasure(ypred_svm_test, ktest_pca(:, end));
acc_svm = mean(double(ypred_svm_test == ktest_pca(:,end))) * 100;
printf("\nF-medida da regressao para a base de teste: %.2f\n", fsvm);
fflush(stdout);

mat_res(5,1) = fsvm;
mat_res(5,2) = precsvm*100;
mat_res(5,3) = revsvm*100;
mat_res(5,4) = acc_svm;
mat_res(5,5) = time_exec;

printf('\nO algoritmo SVM finalizou a execucao. Pressione enter para continuar...\n'); 
pause;