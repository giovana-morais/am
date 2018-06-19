% Grupo 7 - Reconhecimento de Atividades Humanas
pkg load statistics
if(strcmp(computer(), "x86_64-pc-linux-gnu") )
  addpath("./libsvm/libsvm_x86_64-pc-linux-gnu");
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
load("pre_processed.mat");

% passando os dados pelo PCA para reduzir os atributos e utilizar em algoritmos que demandam mais processamento
data_pca = pca(all_data);

% proveniente do 10-fold cross validation utilizado para a escolha de melhor fold
ksize = floor(rows(all_data)/10);

ktest = [];
ktrain = [];
ktest_pca = [];
ktrain_pca = [];

%%% matriz para receber todos os resultados de predicao das hipoteses criadas:
%%% as linhas serao os algoritmos e as colunas serao os resultados na seguinte sequencia - acuracia e tempo de execucao
mat_res = zeros(4, 2);

% knn com melhor k encontrado com grid search ----------------------------------------------------------
k = 1;
printf("\nIniciando execucao do knn com k = %d\n", k);
fflush(stdout);

tic;
for i = 1:rows(ktest)
  ypred_knn_test(i) = knn(ktrain(:,1:end-1), ktrain(:,end), ktest(i,1:end-1), k);
endfor
time_exec = toc;
acc_knn = mean(double(ypred_knn_test == ktest(:,end))) * 100;
printf("\nAcuracia da validacao: %.2f\n", acc_knn);
fflush(stdout);

mat_res(1, 1) = acc_knn;
mat_res(1, 2) = time_exec;

for i = 1:rows(ktrain)
  ypred_knn_train(i) = knn(ktrain(:,1:end-1), ktrain(:,end), ktrain(i,1:end-1), k);
endfor
acc_knn = mean(double(ypred_knn_train == ktrain(:,end))) * 100;
printf("\nAcuracia do treinamento: %.2f\n", acc_knn);
fflush(stdout);

printf("\nKNN finalizou execucao. Pressione enter para continuar...\n");
pause;

% regressao logistica com melhor lambda encontrado com grid search ---------------------------------------------------
lambda = 1;
printf("\nIniciando execucao de regressao logistica com lambda = %d\n", lambda);
fflush(stdout);
tic;
ypred_rl_test = regression(ktrain(:,1:end-1), ktrain(:,end), ktest(:,1:end-1), lambda);
time_exec = toc;
acc_reg = mean(double(ypred_rl_test == ktest(:,end))) * 100;
printf('\nAcuracia da validacao: %.2f\n', acc_reg);
fflush(stdout);

mat_res(2, 1) = acc_reg;
mat_res(2, 2) = time_exec;

ypred_rl_train = regression(ktrain(:,1:end-1), ktrain(:,end), ktrain(:,1:end-1), lambda);
acc_reg = mean(double(ypred_rl_train == ktrain(:,end))) * 100;
printf('\nAcuracia do treinamento: %.2f\n', acc_reg);
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
time_exec = toc;
printf('\nAcuracia da validacao: %.2f\n', acc_svm);
fflush(stdout);

mat_res(4, 1) = acc_svm;
mat_res(4, 2) = time_exec;

[ypred_svm_train, ~, ~] = svm(ktrain_pca(:,1:end-1), ktrain_pca(:,end), ktrain_pca, c, g);
acc_svm = mean(double(ypred_svm_train == ktrain(:,end))) * 100;
printf('\nAcuracia do treinamento: %.2f\n', acc_svm);
fflush(stdout);

printf('\nO algoritmo SVM finalizou a execucao. Pressione enter para continuar...\n'); 
pause;