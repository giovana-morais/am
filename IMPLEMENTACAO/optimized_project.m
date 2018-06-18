% Grupo 7 - Reconhecimento de Atividades Humanas
pkg load statistics

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

%%% matriz para receber todos os resultados de predicao das hipoteses criadas
mat_res = [];

% knn com melhor k encontrado com grid search
k = 1;
printf("\nIniciando execucao do knn com k = %d\n", k);
fflush(stdout);
ac = 0;
tic();
for i = 1:rows(ktest)
  ypred_knn_test = knn(ktrain(:,1:end-1), ktrain(:,end), ktest(i,1:end-1), k);
  if(ypred_knn_test == ktest(i, end))
    ac += 1;
  endif
endfor
toc();
acc_knn = ac/rows(ktest);
printf("\nAcuracia da validacao: %.2f\n", acc_knn);
fflush(stdout);

ac = 0;
for i = 1:rows(ktrain)
  ypred_knn_train = knn(ktrain(:,1:end-1), ktrain(:,end), ktrain(i,1:end-1), k);
  if(ypred_knn_train == ktrain(i, end))
    ac += 1;
  endif
endfor
acc_knn = ac/rows(ktest);
printf("\nAcuracia do treinamento: %.2f\n", acc_knn);
fflush(stdout);

printf("\nKNN finalizou execucao. Pressione enter para continuar...\n");
pause;

% regressao logistica com melhor lambda encontrado com grid search
lambda = 1;
printf("\nIniciando execucao de regressao logistica com lambda = %d\n", lambda);
fflush(stdout);
tic();
ypred_rl_test = regression(ktrain(:,1:end-1), ktrain(:,end), ktest(:,1:end-1), lambda);
toc();
acc_reg = mean(double(ypred_rl_test == ktest(:,end))) * 100;
printf('\nAcuracia da validacao: %.2f\n', acc_reg);
fflush(stdout);

tic();
ypred_rl_train = regression(ktrain(:,1:end-1), ktrain(:,end), ktest(:,1:end-1), lambda);
toc();
acc_reg = mean(double(ypred_rl_train == ktrain(:,end))) * 100;
printf('\nAcuracia do treinamento: %.2f\n', acc_reg);

printf("\nRegressao logistica finalizou execucao. Pressione enter para continuar...\n");
pause;

% svm com melhor c e gamma encontrados com grid search
c = 1;
gamma = 1;
printf('\nIniciando execucao de SVM\n');
fflush(stdout);
tic();
##################### MUDAR A CHAMADA DO SVM PARA RECEBER O C E O GAMMA #########################
[ypred_svm_test, ~, ~] = svm(ktrain_pca(:,1:end-1), ktrain_pca(:,end), ktest_pca, c, g);
acc_svm = mean(double(y_pred == ktest(:,end))) * 100;
toc();
printf('\nAcuracia da validacao: %.2f\n', acc_svm);

##################### MUDAR A CHAMADA DO SVM PARA RECEBER O C E O GAMMA #########################
[ypred_svm_train, ~, ~] = svm(ktrain_pca(:,1:end-1), ktrain_pca(:,end), ktrain_pca, c, g);
acc_svm = mean(double(ypred_svm_train == ktrain(:,end))) * 100;
printf('\nAcuracia do treinamento: %.2f\n', acc_svm);

printf('\nO algoritmo SVM finalizou a execucao. \n'); 
pause;