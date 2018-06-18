function [J grad] = rnaCusto2(nn_params, ...
                             input_layer_size, ...
                             hidden_layer_size, ...
                             hidden_layers, ...
                             num_labels, ...
                             X, y, lambda)
%RNACUSTO Implementa a funcao de custo para a rede neural com duas camadas
%voltada para tarefa de classificacao
%   [J grad] = RNACUSTO(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) calcula o custo e gradiente da rede neural. The
%   Os parametros da rede neural sao colocados no vetor nn_params
%   e precisam ser transformados de volta nas matrizes de peso.
%
%   input_layer_size - tamanho da camada de entrada
%   hidden_layer_size - tamanho da camada oculta
%   num_labels - numero de classes possiveis
%   lambda - parametro de regularizacao
%
%   O vetor grad de retorno contem todas as derivadas parciais
%   da rede neural.
%
 

 
comeco_camada_1 = 1;
fim_camada_1 = hidden_layer_size * (input_layer_size + 1);

comeco_camada_2 = fim_camada_1 + 1;
fim_camada_2 = comeco_camada_2 + hidden_layer_size * (hidden_layer_size + 1) - 1;

comeco_camada_3 = fim_camada_2 + 1;                 
fim_camada_3 =  comeco_camada_3 - 1 + (hidden_layer_size + 1) * num_labels;

Theta1 = reshape(nn_params(comeco_camada_1:fim_camada_1), ...
                 hidden_layer_size, (input_layer_size + 1));
 
Theta2 = reshape(nn_params(comeco_camada_2:fim_camada_2), ...
                 hidden_layer_size, hidden_layer_size + 1);
                 
Theta3 = reshape(nn_params(comeco_camada_3:fim_camada_3),...
                 num_labels, (hidden_layer_size + 1) );
 
% Definindo variaveis uteis
m = size(X, 1);
          
% As variaveis a seguir precisam ser retornadas corretamente
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));
 
% ====================== INSIRA SEU CODIGO AQUI ======================
% Instrucoes: Voce deve completar o codigo a partir daqui 
%               acompanhando os seguintes passos.
%
% (1): Lembre-se de transformar os rotulos Y em vetores com 10 posicoes,
%      onde tera zero em todas posicoes exceto na posicao do rotulo
%
% (2): Execute a etapa de feedforward e coloque o custo na variavel J.
%      Apos terminar, verifique se sua funcao de custo esta correta,
%      comparando com o custo calculado em ex05.m.
%
% (3): Implemente o algoritmo de backpropagation para calcular 
%      os gradientes e alimentar as variaveis Theta1_grad e Theta2_grad.
%      Ao terminar essa etapa, voce pode verificar se sua implementacao 
%      esta correta atraves usando a funcao verificaGradiente.
%
% (4): Implemente a regularizacaoo na funcaoo de custo e gradiente.
%
 
% -------------------------------------------------------------
 
 
######### Parte 1 #############
 
y_matrix = zeros(y,num_labels);
for i = 1:1:size(y,1)
  y_matrix(i,y(i)) = 1;
endfor

###############################
 
######### Parte 2  ############
 


a1 = [ones(m,1), X]; # 5000 x 400
z2 = a1 * Theta1'; #5000x401 * (401x25) = 5000x25

a2 = [ones(m,1), sigmoide(z2)]; # 5000x26
z3 = a2 * Theta2'; # 5000x26 * 26*25 = 5000x25

a3 = [ones(m,1), sigmoide(z3)]; # 5000x26
z4 = a3 * Theta3'; # 5000x26 * 26x10  = 5000x10

hx = sigmoide(z4); # a4;

#J = (1/m) * sum(sum( (-y_matrix).*log(hx) - (1 - y_matrix).*log(1-hx), 2)); #+ lambda*penalty/(2*m);
for i = 1:m
  for k = 1:num_labels
    J +=  -y_matrix(i,k) * log(hx(i,k)) - ( 1 - y_matrix(i,k)) * log(1-hx(i,k));
  endfor
endfor  

J /= m;
  
######### Parte 3 ############ 


# calculando quem foram os culpados pra previsao dar errada....

#comecando do final da nn:
d4 = hx - y_matrix; # 5000x401
size(d4)
d3 = d4 * Theta3(:, 2:end) .* gradienteSigmoide(z3); #400x5000 * 5000x10 === 400x10
d2 = d3 * Theta2(:, 2:end) .* gradienteSigmoide(z2);   #400x10 

 
Delta3 = d4' * a3;
Delta2 = d3' * a2;
Delta1 = d2' * a1;



# Desconsiderando os bias dos pesos para realizar o recalculo
Theta1_sembias = [zeros(size(Theta1,1),1), Theta1(:,2:end)];
Theta2_sembias = [zeros(size(Theta2,1),1), Theta2(:,2:end)];
Theta3_sembias = [zeros(size(Theta3,1),1), Theta3(:,2:end)];

# Valor final dos Thetas gradientes:
Theta1_grad = Delta1/m + lambda/m * Theta1_sembias;
Theta2_grad = Delta2/m + lambda/m * Theta2_sembias;
Theta3_grad = Delta3/m + lambda/m * Theta3_sembias;
 
###############################

######### Parte 4 #############

#Calculando a regularizacao, conforme pagina 4 do pdf, lembrando que eh preciso desconsiderar o bias !!

r = sum(sumsq(Theta1(:,2:end), 1)) + sum(sumsq(Theta2(:,2:end), 1));
r *= lambda/(2*m);
J += r;
     
###############################
 
% =========================================================================
 
% Juntando os gradientes
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];
 #}
end
