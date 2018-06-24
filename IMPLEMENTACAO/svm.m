function [ypred, gridLin, gridRbf] = svm(train, train_labels, test, c, g) 

    
    % se ja recebeu c e g, nao precisa incrementar parametros    
    if(exist("c", "var") && exist("g", "var"))
      
      gridLin = [];
      gridRbf = [];
      
      c = num2str(c);
      g = num2str(g);
    
      parametersRBF = ["-c " c " -t 2 -g " g " -q"];
      
      modelrbf = svmtrain(train_labels, train, ...
                         parametersRBF);
                         
      ypred = svmpredict(test(:, end), test(:,1:end-1), modelrbf);

      
    else

      i = 2^(-5);
      iterC = 0;
      iterGamma = 0;
      
      gridLin = zeros(1, 8);
      gridRbf = zeros(1, 152);
      
      % Laço para iterar o C
      % O C default eh 1, com seu range indo de 0 a infinito.
      % Irei iterar o C de 2^-3 ate 2^4
      % Posso realizar o grid dessa maneira pois C e gamma sao independentes
      while (i <= (2^4))
        %i = 16; %tirar isso  
        c = num2str(i);
        
        j = 2^(-15);
        iterC = iterC + 1;
        % Laço para iterar gamma (parametro do kernel).
        % O default eh 1/71 (numero de features)
        % vou iterar o gamma de 2^-15 ate 2^3
        while (j <= (2^3))
          iterGamma = iterGamma + 1;
          % j = 16; %tirar isso
          g = num2str(j);
        
          parametersRBF = ["-c " c " -t 2 -g " g " -q"];
          
          % Kernel rbf
          modelrbf = svmtrain(train_labels, train, ...
                         parametersRBF);

          % Predições do RBF       
          [y_predforRBF, acuraciaRBF, dec_values_R] = svmpredict(test(:, end), test(:,1:end-1), modelrbf);
          
          fsvm = fmeasure(y_predforRBF, test(:,end));
          
          %gridRbf(1, iterGamma) = acuraciaRBF(1);
          gridRbf(1, iterGamma) = fsvm;
          
          j = j * 2
          i
          fflush(stdout);
          
        endwhile
        
        parametersLinear = ["-c " c " -t 0 -q"];
          
        %kernel Linear
        modelLinear = svmtrain(train_labels, train, ...
                       parametersLinear);
        
        [y_predforLinear, acuraciaLinear, dec_values_L] = svmpredict(test(:, end), test(:,1:end-1), modelLinear);
        
        fsvm = fmeasure(y_predforLinear, test(:,end));
        
        %gridLin(1, iterC) = acuraciaLinear(1);
        gridLin(1, iterC) = fsvm;
      
        
        i = i * 2;
        
      endwhile
      
  endif
  
  %{
  
  -s svm_type : set type of SVM (default 0)
          0 -- C-SVC              (multi-class classification)
          1 -- nu-SVC             (multi-class classification)
          2 -- one-class SVM
          3 -- epsilon-SVR        (regression)
          4 -- nu-SVR             (regression)
  -t kernel_type : set type of kernel function (default 2)
          0 -- linear: u'*v
          1 -- polynomial: (gamma*u'*v + coef0)^degree
          2 -- radial basis function: exp(-gamma*|u-v|^2)
          3 -- sigmoid: tanh(gamma*u'*v + coef0)
          4 -- precomputed kernel (kernel values in training_instance_matrix)
  -d degree : set degree in kernel function (default 3)
  -g gamma : set gamma in kernel function (default 1/num_features)
  -r coef0 : set coef0 in kernel function (default 0)
  -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
  -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
  -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
  -m cachesize : set cache memory size in MB (default 100)
  -e epsilon : set tolerance of termination criterion (default 0.001)
  -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
  -b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
  -wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
  -v n : n-fold cross validation mode
  -q : quiet mode (no outputs)
  
  %}
  
end
