function [f, prec, recall] = fmeasure(ypred, yreal)
  if columns(ypred) > 1
    ypred = ypred';
  endif
  if columns(yreal) > 1
    yreal = yreal';
  endif
  
  for i = 1:6
    tp = sum(ypred == i & yreal == i);
    fp = sum(ypred == i & yreal != i);
    tn = sum(ypred != i & yreal == i);
    if tp + fp > 0
      % precisao para cada classe
      prec(i) = tp/(tp+fp);
    endif
    if tp + tn > 0
      % revocacao para cada classe
      recall(i) = tp/(tp+tn);
    endif
  end
  
  prec = mean(prec);
  recall = mean(recall);
  
  % calculo da f-medida com a media da precisao e revocacao
  f = (2 * prec * recall / (prec + recall)) * 100;
end