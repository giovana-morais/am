function [x, y, z] =  r1_to_r3(i, maxx, maxy, maxz)
  # Mapeia indices de vetores para uma matriz 3D
  # No caso das redes neurais:
  # x eh a norma do vetor do numero de iteracoes
  # y eh a norma do vetor lambda
  # z eh a norma do vetor neurons_size
  
  z = ceil(i / (maxx * maxy));
  y = mod( ceil(i /maxx) , maxy);
  x = mod(i , maxx); 
  if (x==0)  x = maxx; endif
  if (y==0)  y = maxy; endif
  if (z==0)  z = maxz; endif
endfunction