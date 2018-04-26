function [pAtrVitoria, pAtrDerrota] = calcularProbabilidades(X, Y)
%CALCULARPROBABILIDADES Computa a probabilidade de ocorrencia de cada 
%atributo por rotulo possivel. A funcao retorna dois vetores de tamanho n
%(qtde de atributos), um para cada classe.
%   [pAtrVitoria, pAtrDerrota] = CALCULARPROBABILIDADES(X, Y) calcula a 
%   probabilidade de ocorrencia de cada atributo em cada classe. 
%   Cada vetor de saida tem dimensao (n x 1), sendo n a quantidade de 
%   atributos por amostra.

% inicializa os vetores de probabilidades
pAtrVitoria = zeros(size(X,2),1);
pAtrDerrota = zeros(size(X,2),1);

pVitoria = sum(Y==1)/size(Y,1); 
pDerrota = sum(Y==0)/size(Y,1);

% ====================== ESCREVA O SEU CODIGO AQUI ======================
% Instrucoes: Complete o codigo para encontrar a probabilidade de
%               ocorrencia de um atributo para uma determinada classe.
%               Ex.: para a classe 1 (vitoria), devera ser computada um
%               vetor pAtrVitoria (n x 1) contendo n valores:
%               P(Atributo1=1|Classe=1), ..., P(Atributo5=1|Classe=1), e o
%               mesmo para a classe 0 (derrota):
%               P(Atributo1=1|Classe=0), ..., P(Atributo5=1|Classe=0).
%
  
  for j = 1: size(X,2)    
    atrVit = 0;
    atrDer = 0;

    for i = 1: size(X,1)
      if X(i,j) == 1 && Y(i) == 1
        atrVit++;  
      end
      if X(i,j) == 0 && Y(i) == 0
        atrDer++;
      end
      pAtrVitoria(j) = atrVit/size(X,1);
      pAtrDerrota(j) = atrDer/size(X,1);
    end
  end
  
  pAtrVitoria /= pVitoria;
  pAtrDerrota /= pDerrota;

% =========================================================================

end