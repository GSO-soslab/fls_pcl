clc
clear

f = 1.2*1000*1000;
c= 1500;
lambda = c/f;
k = 2*pi/lambda;
h= 0.0065;

D = [];
for a = -6:0.1:6
    temp = k*h/2*sind(a);
    DI = sin(temp)/temp;
    D = [D DI];

end

plot(-6:0.1:6, D,'-x')