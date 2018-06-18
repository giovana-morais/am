function R = leakly_relu(z)
  R = max(0.01*y,z);
endfunction