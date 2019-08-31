%% LSTM  批处理下LSTM
clear
clc
close all
%% 加载数据
clear;clc;close all;format compact
%% 加载数据
qx1=xlsread('沧州气象日度数据.xlsx','B2:G362');%由于有缺失值，因此只读了前几列和最后几列，中间几列没有读取
qx2=xlsread('沧州气象日度数据.xlsx','J2:O362');
qx=[qx1 qx2];
wr=xlsread('沧州污染日度数据.xlsx','C2:C362');
input=[wr(1:end-1,:) qx(2:end,:)];%输入为前一天的pm2.5+预测日的气象  输出为预测日的pm2.5
output=wr(2:end,:);
P=mapminmax(input',0,1);
[T,outputns]=mapminmax(output',0,1);

%% 提取300个样本为训练样本，剩下样本为预测样本

n=1:size(P,2);
i=300;
train_data=P(:,n(1:i));
train_label=T(:,n(1:i),:);
P_test=P(:,n(i+1:end));
T_test=T(:,n(i+1:end));

data_length=size(train_data,1);
% 采用minibatch进行网络训练 首先需要对训练数据进行分批
%分numbatches批 每批data_num个 共numbatches*data_num个数据
numbatches=3;
data_num=100;
for i=1:numbatches
    train1=train_data(:,(i-1)*data_num+1:i*data_num);
    batchdata(:,:,i)=train1;
end

for i=1:numbatches
    train2=train_label(:,(i-1)*data_num+1:i*data_num);
    batchdata_target(:,:,i)=train2;
end
%% 网络参数初始化
% 结点数设置
input_num=data_length;%输入层节点
cell_num=5;%隐含层节点
output_num=1;%输出层节点
dropout=0;%dropout系数
cost_gate=1e-10;% 误差要求精度
% ab=20;
ab=4*sqrt(6/(cell_num+output_num));%  利用均匀分布进行初始化
% ab=1/ab;

%% 网络初始化
% 网络偏置初始化
bias_input_gate=rand(1,cell_num);
bias_forget_gate=rand(1,cell_num);
bias_output_gate=rand(1,cell_num);
% 网络权重初始化
weight_input_x=rand(input_num,cell_num)/ab;
weight_input_h=rand(output_num,cell_num)/ab;
weight_inputgate_x=rand(input_num,cell_num)/ab;
weight_inputgate_c=rand(cell_num,cell_num)/ab;
weight_forgetgate_x=rand(input_num,cell_num)/ab;
weight_forgetgate_c=rand(cell_num,cell_num)/ab;
weight_outputgate_x=rand(input_num,cell_num)/ab;
weight_outputgate_c=rand(cell_num,cell_num)/ab;
% hidden_output权重
weight_preh_h=rand(cell_num,output_num);
% 网络状态初始化
h_state=rand(output_num,data_num);
cell_state=rand(cell_num,data_num);
%% 网络训练学习
for iter=1:100%训练次数
    yita=1/(10+sqrt(iter)); %自适应学习率
    for n=1:numbatches%利用minibatch方式进行训练
        train_data=batchdata(:,:,n);
        train_label=batchdata_target(:,:,n);
    for m=1:data_num
        %前馈部分
        if(m==1)
            gate=tanh(train_data(:,m)'*weight_input_x);
            input_gate_input=train_data(:,m)'*weight_inputgate_x+bias_input_gate;
            output_gate_input=train_data(:,m)'*weight_outputgate_x+bias_output_gate;
            for n=1:cell_num
                input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
                output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
            end
            forget_gate=zeros(1,cell_num);
            forget_gate_input=zeros(1,cell_num);
            cell_state(:,m)=(input_gate.*gate)';
        else
            gate=tanh(train_data(:,m)'*weight_input_x+h_state(:,m-1)'*weight_input_h);
            input_gate_input=train_data(:,m)'*weight_inputgate_x+cell_state(:,m-1)'*weight_inputgate_c+bias_input_gate;
            forget_gate_input=train_data(:,m)'*weight_forgetgate_x+cell_state(:,m-1)'*weight_forgetgate_c+bias_forget_gate;
            output_gate_input=train_data(:,m)'*weight_outputgate_x+cell_state(:,m-1)'*weight_outputgate_c+bias_output_gate;
            for n=1:cell_num
                input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
                forget_gate(1,n)=1/(1+exp(-forget_gate_input(1,n)));
                output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
            end
            cell_state(:,m)=(input_gate.*gate+cell_state(:,m-1)'.*forget_gate)';   
        end
        pre_h_state=tanh(cell_state(:,m)').*output_gate;
        h_state(:,m)=(pre_h_state*weight_preh_h)';
        %误差计算
        Error=h_state(:,m)-train_label(:,m);
        Error_Cost(1,iter)=sum(Error.^2);
        if(Error_Cost(1,iter)<cost_gate)
            flag=1;
            break;
        else %权重更新
            [   weight_input_x,...
                weight_input_h,...
                weight_inputgate_x,...
                weight_inputgate_c,...
                weight_forgetgate_x,...
                weight_forgetgate_c,...
                weight_outputgate_x,...
                weight_outputgate_c,...
                weight_preh_h ]=LSTM_updata_weight(cell_num,output_num,m,yita,Error,...
                                                   weight_input_x,...
                                                   weight_input_h,...
                                                   weight_inputgate_x,...
                                                   weight_inputgate_c,...
                                                   weight_forgetgate_x,...
                                                   weight_forgetgate_c,...
                                                   weight_outputgate_x,...
                                                   weight_outputgate_c,...
                                                   weight_preh_h,...
                                                   cell_state,h_state,...
                                                   input_gate,forget_gate,...
                                                   output_gate,gate,...
                                                   train_data,pre_h_state,...
                                                   input_gate_input,...
                                                   output_gate_input,...
                                                   forget_gate_input);
                
        end
    end
    if(dropout>0) %Dropout
        rand('seed',0)
           weight_inputgate_x =weight_inputgate_x.*(rand(size(weight_inputgate_x))>dropout);
    end
       
    if(Error_Cost(1,iter)<cost_gate)
        break;
    end
    end
end
%% 测试阶段
%数据加载
for i=1:size(P_test,2)
test_final=P_test(:,i);
test_output=T_test(:,i);
%前馈
m=2;
gate=tanh(test_final'*weight_input_x+h_state(:,m-1)'*weight_input_h);
input_gate_input=test_final'*weight_inputgate_x+cell_state(:,m-1)'*weight_inputgate_c+bias_input_gate;
forget_gate_input=test_final'*weight_forgetgate_x+cell_state(:,m-1)'*weight_forgetgate_c+bias_forget_gate;
output_gate_input=test_final'*weight_outputgate_x+cell_state(:,m-1)'*weight_outputgate_c+bias_output_gate;
for n=1:cell_num
    input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
    forget_gate(1,n)=1/(1+exp(-forget_gate_input(1,n)));
    output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
end
cell_state_test=(input_gate.*gate+cell_state(:,m-1)'.*forget_gate)';
pre_h_state=tanh(cell_state_test').*output_gate;
h_state_test=(pre_h_state*weight_preh_h)';

sim(:,i)=h_state_test;
end

%反归一化
test_sim=mapminmax('reverse',sim,outputns);%网络预测数据
test=mapminmax('reverse',T_test,outputns);%实际数据

%%
%%
figure
plot(test_sim,'-bo');
grid on
hold on
plot(test,'-r*');
legend('预测数据','实际数据')
title('LSTM神经网络回归预测')
xlabel('样本数')
ylabel('PM2.5含量')
% 相对误差
figure
bar(abs(test-test_sim)./test)
title('LSTM')
ylabel('相对误差')
xlabel('样本数')
% 平均相对误差
pjxd=sum(abs(test-test_sim)./test)/length(test)