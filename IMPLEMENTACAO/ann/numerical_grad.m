function numgrad = numerical_grad(J, theta)
  % Calcula o gradiente usando "diferencas finitas"
  %e da um resultado estimado do gradiente.
  
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