function numgrad = gradienteNumerico(J, theta)
%GRADIENTENUMERICO Calcula o gradiente usando "diferencas finitas"
%e da um resultado estimado do gradiente.
%   numgrad = GRADIENTENUMERICO(J, theta) calcula o gradiente numerico
%   da funcao J acerca do theta. Executando y = J(theta) deve
%   retorna o valor da funcao para theta.

% Notas: O codigo a seguir implementa a checagem do gradiente numerico
%        e retorna o gradiente numerico. O valor numgrad(i) se refere
%        a uma aproximacao numerico da derivada parcial de J com relacao
%        ao i-esimo argumento de entrada. Ou seja, numgrad(i) se
%        refere a derivada parcial aproximada de J com relacao a theta(i).
%                

numgrad = zeros(size(theta));
perturb = zeros(size(theta));
e = 1e-4;

for p = 1:numel(theta)

    % Define vetor de perturbacao
    perturb(p) = e;
    loss1 = J(theta - perturb);
    loss2 = J(theta + perturb);
	
    % Calcula o gradiente numerico
    numgrad(p) = (loss2 - loss1) / (2*e);
    perturb(p) = 0;
	
end

end
