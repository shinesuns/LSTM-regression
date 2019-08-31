%% 清空环境变量 elm
clc
clear
close all
format compact
%% 导入数据
load data
load outputps
P_train=X_train;
T_train=Y_train;
P_test=X_test;
T_test=Y_test;
%% 提取300个样本为训练样本，剩下样本为预测样本

%% 节点个数
inputnum=size(P_train,1);%输入层节点
hiddennum=2; %隐含层节点
type='sig';%sin%hardlim%sig%隐含层激活函数
TYPE=0;%0=回归  1=分类
[IW,B,LW,TF,TYPE] = elmtrain(P_train,T_train,hiddennum,type,TYPE);
%% ELM仿真测试
sim = elmpredict(P_test,IW,B,LW,TF,TYPE);
test_sim=mapminmax('reverse',sim,outputps);%网络预测数据
% test=mapminmax('reverse',T_test,outputps);%实际数据
test=T_test;%实际数据

%
figure
plot(test_sim,'-bo');
grid on
hold on
plot(test,'-r*');
legend('预测数据','实际数据')
title('ELM神经网络回归预测')
xlabel('样本数')
ylabel('PM2.5含量')
% 相对误差
figure
plot(abs(test-test_sim)./test)
title('ELM')
ylabel('相对误差')
xlabel('样本数')
% 平均相对误差
pjxd=sum(abs(test-test_sim)./test)/length(test)