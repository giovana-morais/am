function g = gradienteSigmoide(z)
%GRADIENTESIGMOIDE retorna o gradiente da funcao sigmoidal para z
%
%   g = GRADIENTESIGMOIDE(z) calcula o gradiente da funcao sigmoidal
%   para z. A funcao deve funcionar independente se z for matriz ou vetor.
%   Nestes casos,  o gradiente deve ser calculado para cada elemento.

g = zeros(size(z));

% ====================== INSIRA SEU CODIGO AQUI ======================
% Instrucoes: Calcula o gradiente da funcao sigmoidal para 
%               cada valor de z (seja z matriz, escalar ou vetor).


g = sigmoide(z) .* (1 - sigmoide(z));

% =============================================================

end
