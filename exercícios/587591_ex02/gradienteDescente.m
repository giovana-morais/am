function [theta, J_historico] = gradienteDescente(X, y, theta, alpha, num_iter)
%GRADIENTEDESCENTE Executa o gradiente descente para aprender e otimizar theta
%   theta = GRADIENTEDESENTE(X, y, theta, alpha, num_iter) atualiza theta usando 
%   num_iter passos do gradiente com taxa de aprendizado alpha

% Initializa algumas variaveis uteis
m = length(y); % numero de exemplos de treinamento
J_historico = zeros(num_iter, 1);

for iter = 1:num_iter

    % ====================== ESCREVA O SEU CODIGO AQUI ====================
    % Instrucoes: Execute um unico passo do gradiente para ajustar o vetor
    %             theta. 
    %
    % Dica: para verificar se a o gradiente esta correto, verifique se a 
    %       funcao de custo (computarCusto) nunca aumenta de valor no 
    %       decorrer das iteracoes. Para facilitar, em ex02.m ha uma funcao
    %       que plota o custo J no decorrer das iteracoes. A linha nunca
    %       pode ser crescente. Se for, reduza alpha.
    %
    
    sum_1 = 0;
    sum_2 = 0;
    
    % sum_{1}^{m} h_{theta}(x^{i} - y^{i})*x^{i}_{j}
    for i = 1: m
      sum_1 = sum_1 + ((theta(1) + theta(2) * X(i,2)) - y(i));
      sum_2 = sum_2 + (((theta(1) + theta(2) * X(i,2)) - y(i)) * X(i,2));
    endfor 
    
    theta_1 = theta(1) - (alpha/m) * sum_1;
    theta_2 = theta(2) - (alpha/m) * sum_2;
      
    theta(1) = theta_1;
    theta(2) = theta_2;
    % ============================================================

    % Armazena o custo J obtido em cada iteracao do gradiente 
    J_historico(iter) = computarCusto(X, y, theta);

end

end
