function [J grad] = rnaCusto(nn_params, ...
                             input_layer_size, ...
                             hidden_layer_size, ...
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
 
% Extrai os parametros de nn_params e alimenta as variaveis Theta1 e Theta2.
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
 
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
 
% Definindo variaveis uteis
m = size(X, 1);
          
% As variaveis a seguir precisam ser retornadas corretamente
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
 
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
 
# Primeiro e necessario adicionar o bias ao vetor de entrada, para realizar a multiplicacao de matrizes corretamente.
input_primeira_camada = zeros(size(X,1) , size(X,2) + 1);
for i = 1:size(X,1)
  input_primeira_camada(i,:) = [1, X(i,:)];
endfor
 
# Calculando os valores das entradas aplicando os pesos  
valores_primeira_camada = input_primeira_camada * Theta1';
 
# Utilizando a funcao de ativacao da primeira camada
# Novamente, adicionando o bias agora no input da camada intermediaria (que eh o output da primaria)

for i = 1:size(valores_primeira_camada,1)
  input_segunda_camada(i,:) = [1, sigmoide(valores_primeira_camada(i,:))];
endfor

 
# Aplicando os pesos ao output da camada intermediaria, (e input da camada final)
input_terceira_camada = input_segunda_camada * Theta2';
# realizando  o processo
hx = sigmoide(input_terceira_camada);

#J = (1/m) * sum(sum( (-y_matrix).*log(hx) - (1 - y_matrix).*log(1-hx), 2)); #+ lambda*penalty/(2*m);
for i = 1:size(X,1)
  for k = 1:num_labels
    J +=  -y_matrix(i,k) * log(hx(i,k)) - ( 1 - y_matrix(i,k)) * log(1-hx(i,k));
  endfor
endfor  

J /= m;
  
######### Parte 3 ############


# calculando quem foram os culpados pra previsao dar errada....

#comecando do final da nn:
delta_final = hx - y_matrix;

#indo agora calcular os erros para a camada "oculta" conforme a ultima equacao da pag 5 do pdf.
delta_hidden = delta_final * Theta2(:, 2:end) .* gradienteSigmoide(valores_primeira_camada);
 
# Para a camada oculta:
Delta1 = delta_hidden' * input_primeira_camada;

# Para a camada final:
Delta2 = delta_final' * input_segunda_camada;


# Desconsiderando os bias dos pesos para realizar o recalculo
Theta1_sembias = [zeros(size(Theta1,1),1), Theta1(:,2:end)];
Theta2_sembias = [zeros(size(Theta2,1),1), Theta2(:,2:end)];

# Valor final dos Thetas gradientes:
Theta1_grad = Delta1/m + lambda/m * Theta1_sembias;
Theta2_grad = Delta2/m + lambda/m * Theta2_sembias;
 
 
###############################

######### Parte 4 #############

#Calculando a regularizacao, conforme pagina 4 do pdf, lembrando que eh preciso desconsiderar o bias !!

r = sum(sumsq(Theta1(:,2:end), 1)) + sum(sumsq(Theta2(:,2:end), 1));
r *= lambda/(2*m);
J += r;
     
###############################
 
% =========================================================================
 
% Juntando os gradientes
grad = [Theta1_grad(:) ; Theta2_grad(:)];
 
end