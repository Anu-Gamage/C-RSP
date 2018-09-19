% Test iterative algorithm for FE 
clear;clc;close all;
n = 300;
k = 2;
m = 2;
cin = 10;
lambda = 0.9;
[A,labels] = mlsbm_gen(n,k,m,cin,lambda);

b = 0.02;
dFE = CFE(A,n,k,m,b,labels);

load('P.mat'); load('C.mat')
dFE_iter = fe_iterative(P_joint,C_joint, b, dFE);


