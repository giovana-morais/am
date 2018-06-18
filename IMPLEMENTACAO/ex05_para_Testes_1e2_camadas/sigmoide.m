function g = sigmoide(z)
%SIGMOIDE Calcula a sigmoide dado um valor para z
%   J = SIGMOID(z) calcula a sigmoide dado um valor para z

g = 1.0 ./ (1.0 + exp(-z));

end
