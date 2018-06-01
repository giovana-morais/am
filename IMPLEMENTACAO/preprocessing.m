#!/usr/bin/octave -qf
% pre-processamento
printf("Carregando dados...\n\n");
fflush(stdout);

% leitura dos dados de treinamento
train_data = load("-ascii", "dataset_uci/final_X_train.txt");

% leitura dos dados de teste
test_data = load("-ascii", "dataset_uci/final_X_test.txt");

% mantem todos os dados de X em apenas uma matriz
X_data = [train_data; test_data];
	
% leitura das classes de treinamento
train_classes = load("-ascii", "dataset_uci/final_y_train.txt");

% leitura das classes de teste
test_classes = load("-ascii", "dataset_uci/final_y_test.txt");

% mantem todos os dados de Y em apenas uma matriz
Y_data = [train_classes; test_classes];

printf("\nDados carregados!\n\nRemovendo dados redundantes da base de dados...\n");
fflush(stdout);


% concatena matriz de amostra X com suas classes
all_data = [X_data, Y_data];
printf("Tamanho antes de remover pesos: %d\n", length(all_data));

# removendo peso na mesma classe
all_data = unique(all_data, "rows");
printf("Tamanho apos remover pesos: %d\n", length(all_data));


if( length(all_data) != length(unique(X_data, "rows")) ) ## SE HOUVER AMOSTRAS COM X IGUAL E Y DIFERENTE :::::::
#### Algoritmo que utilizamos para verificar a existencia de dados espurios (Mesmo valores de x com y diferente)#######
    
    #Primeiramente agrupamos as amostras de acordo com as classes:
    indices = cell(1,6);
    for ind = 1:6
      indices(ind) = find ( all_data (:, end) == ind);
    endfor

    # Criando armazenamento dos indices pra apagar:
    apagar = cell(1,6);


    for conjA = 1:5
        for conjB = conjA+1:6  #testando todas as permutações de conjuntos
            [elements, ind_conjA_pra_apagar, ind_conB_pra_apagar] = intersect ( all_data ( cell2mat(indices( conjA)) , 1:end-1) , all_data ( cell2mat(indices( conjB)) , 1:end-1), "rows" );
            if(length(ind_conjA_pra_apagar) > 0)
		# Extendendo os conjuntos de repetição
                apagar(conjA) = union(cell2mat(apagar(conjA)), ind_conjA_pra_apagar);
                apagar(conjB) = union(cell2mat(apagar(conjB)), ind_conjB_pra_apagar);

            endif  
        endfor
    endfor
    
    # Removendo as tuplas espurias...
    for ind = 1:6
        all_data( cell2mat(apagar(ind)) , : ) = [];
    endfor

endif ### fim da remocao de elementos com X igual porem Y diferente


% codigo para reducao de outliers atraves de percentis
for i = 1:(columns(all_data)-1)
  column = all_data(:,i);
  q1 = prctile(column, 25);
  q3 = prctile(column, 75);
  
  % encontra onde estao os valores maiores e menores que o 3o e o 1o percentil
  gt = find(column > q3);
  lt = find(column < q1);
  
  % substitui valores
  all_data(gt, i) = q3;
  all_data(lt, i) = q1;
end

% Implementacao previa da correlacao entre atributos
#{
printf("\nCalculando matriz de correlacao dos atributos...\n");
fflush(stdout);

% calcula correlacoes entre pares de atributos (menos a coluna de classes)
R = corr(all_data(:,1:end-1));

printf("\nCorrelacoes calculadas\n");

% retira a diagonal principal e o triangulo superior
R = tril(R, -1);

printf("\nDeletando colunas com indice de correlacao muito alto...\n");
fflush(stdout);
delete = [];
% percorre apenas triangulo inferior da matriz de correlacoes
for i = 2:rows(R)
	for j = 1:i
    % verifica se a correlacao eh maior que 0.9 ou menor que -0.9 e verifica se coluna ja nao foi deletada
		if((R(i,j) >= .9 || R(i,j) <= -0.9) && ~ismember(j, delete))
       % verifica as colunas identicas
       if(R(i,j) == 1)
        printf("\nAs colunas %d e %d sao identicas, ou seja, tem correlacao = 1\n", i, j);
       endif
      % insere na lista de colunas a serem deletadas se satisfaz a condicao
      delete = [delete; j];
		endif
	end
end

% pega colunas que nao serao deletadas para checagem
notdeleted = setdiff([1:columns(all_data)], delete);

% deleta as colunas de all_data com base nas correlacoes encontradas
%all_data(:, delete) = [];
printf("\nDe %d colunas antes do calculo de correlacoes, agora restaram %d colunas de atributos:\n", (length(R)+1), columns(all_data));
printf("\n%d %d %d %d %d %d %d %d %d", notdeleted);
printf("\n");
#}

% para rodar a correlacao, a linha 81 deve ser descomentada
all_data = pca(all_data);

% para checagem de balanceamento de dados
printf("\nPara checarmos o balanceamento dos dados atuais, temos que:\n");
printf("Para Y = 1, existem %d amostras\n", length(find(all_data(:,end) == 1)));
printf("Para Y = 2, existem %d amostras\n", length(find(all_data(:,end) == 2)));
printf("Para Y = 3, existem %d amostras\n", length(find(all_data(:,end) == 3)));
printf("Para Y = 4, existem %d amostras\n", length(find(all_data(:,end) == 4)));
printf("Para Y = 5, existem %d amostras\n", length(find(all_data(:,end) == 5)));
printf("Para Y = 6, existem %d amostras\n", length(find(all_data(:,end) == 6)));

% fazemos a normalizacao dos dados para o uso dos classificadores
% o resultado das hip�teses pode ser influenciada pela escala dos atributos
% aqui vamos normalizar para media = 0 e desvio padrao = 1

% media
m = mean(all_data(:,1:end-1));
  
% desvio padrao
s = std(all_data(:,1:end-1));
  
% calcula a norma de cada amostra
% data_norm = (all_data(:,1:end-1)- m)./s;

% concatena dados normalizados com a coluna de classes (Y)
% all_data = [data_norm, all_data(:,end)];


% gera a matriz aleatoria pra facilitar na hora de fazer o cross-fold
temp = all_data(randperm(length(all_data)),:);
all_data = temp;

% salva a matriz ja pre-processada no arquivo binario "pre_processed"
save("pre_processed.mat", "all_data");
printf("\nDados salvos em pre_processed\n\n");
