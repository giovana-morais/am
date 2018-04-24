  function [classe, probVitoria, probDerrota] = classificacao(x,pVitoria,...
                                          pDerrota,pAtrVitoria,pAtrDerrota)
%CLASSIFICACAO Classifica se a entrada x pertence a classe 0 ou 1 usando
%as probabilidades extraidas da base de treinamento.
%   [classe, probVitoria, probDerrota] = CLASSIFICACAO(x,pVitoria,pDerrota,
%                                                  pAtrVitoria,pAtrDerrota) 
%   estima a predicao de x atraves da maior probabilidade da amostra  
%   pertencer a classe 1 ou 0. Tambem retorna as probabilidades condicionais
%   de vitoria e derrota, respectivamente. 

% inicializa a classe e as probabilidades condicionais
classe = 0;
probVitoria = 0;
probDerrota = 0;
	
% ====================== ESCREVA O SEU CODIGO AQUI ======================
% Instrucoes: Complete o codigo para estimar a classificacao da amostra
%               usando as probabilidades extraidas da base de treinamento.
%               Voce precisara encontrar as probabilidades Bayesianas 
%               probVitoria p(classe=1|x) e probDerrota p(classe=0|x) e
%               selecionar a maior.
%
  prodVit = 1;
  prodDer = 1;

  for i = 1: size(x,1)
    if x(i) == 1
      prodVit *= pAtrVitoria(i); 
      prodDer *= (1 - pAtrDerrota(i));
    else
      prodVit *= (1 - pAtrVitoria(i));
      prodDer *= pAtrDerrota(i);      
    end
  endfor

  probVitoria = pVitoria * prodVit;
  probDerrota = pDerrota * prodDer;  
  
  if probVitoria >= probDerrota
    classe = 1;
  end     
% ========================================================================= 

end