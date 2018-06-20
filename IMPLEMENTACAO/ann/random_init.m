function W = random_init(il_size, hl_size)
  % inicializa os pesos randomicamente
  % il_size : tamanho da camada de input
  % hl_size : tamanho da camada oculta
  
  % retorna uma matriz de tamanho il_size X hl_size
  % pra funcionar, tem que instalar dois pacotes
  %   pkg install -forge io
  % e depois
  %   pkg install -forge statistics
  W = 0;
  while( any(any( W == 0 )) )
    W = stdnormal_rnd(hl_size, il_size+1);
  endwhile  
  % W = randn(il_size, hl_size);
endfunction