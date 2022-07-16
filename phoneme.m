function f = phoneme(s)

[val,m_b] = min(s);

if m_b == 1 || m_b == size(s,1)
  s2 = s([1,size(s,1)]);
  sd = diff(s2);
  if m_b == 1
      h_l = 0;
      h_r = sd(1);
  else
      h_l = sd(1);
      h_r = 0;
  end
else
  s2 = s([1,m_b,size(s,1)]);
  sd = diff(s2);
  h_l = sd(1);
  h_r = sd(2);
end

for i=1:size(s,1)
   if i <= m_b
       h(i) = h_l;
       m(i) = -1*h_l*s(m_b);
   else
       h(i) = h_r;
       m(i) = h_r*s(m_b);
   end
   
   f(i) = h(i)*s(i)+m(i);
   
end
    
end
