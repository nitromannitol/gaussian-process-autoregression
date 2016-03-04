##############################################################################
# This code generates all the figures in the paper Gaussian process regresion.  
#
# This material was developed as a final project for Stats 370
# Bayesian Statistics
##############################################################################
using JSON 
using DataStructures
using PyPlot
using Distributions

#include all the helper functions
include("anyOrderFunctions.jl")

###############
###
###  In this section we read in, process the data, and do preliminary plots
###
###############


##Read in data
data = JSON.parsefile("dataset/close.json", dicttype=DataStructures.OrderedDict);
data = data["bpi"];

closingVals = Float64[]; 
for dataPoint in data
	push!(closingVals, dataPoint[2]);
end
T = length(closingVals);



#Do analysis of the data
closingVals = log10(closingVals);

#Plot price of bitcoin over past 5 years
figure();
ax = gca();
ax[:set_xlim]([0,T])
plot(closingVals);
xlabel("Days");
ylabel("log(Price)");
savefig("figures/bitcoin_prices.eps")

#Plot price of bitcoin over 50 days
figure();
plot(closingVals)
ax = gca();
ax[:set_xlim]([400,450])
ax[:set_ylim]([0.5,1.2])
xlabel("Days");
ylabel("log(Price)");
savefig("figures/bitcoin_prices_zoomed.eps")


#create test and train matrices for memory window p
p = 10;


#train on first 4 years and test on last year
n = floor(T*(4/5));
trainInd = n;
trainData = closingVals[1:trainInd];
testData = closingVals[trainInd+1:end];

#normalize training data by second moment
trainData = trainData./(sum(trainData.^2)/length(trainData));

A_train = toeplitz(trainData, p)[p:(end-p),:];
y_train = trainData[(p+1):end];

A_test = toeplitz(testData,p)[p:(end-p),:]
y_test = testData[(p+1):end];

#baseline RMS for train and test
norm(y_train[1:end-1] - y_train[2:end])^2
#0.129

norm(y_test[1:end-1] - y_test[2:end])^2
#0.035


###############
###
###  In this section we optimize the hyperparameters
###
###############


n,m = size(A_train);
X = zeros(n*m, n); 

for j in 1:m
	Xj = A_train[:,j];
	Xj = diagm(Xj);
	loc = n*(j-1)+1:(n*j);
	X[loc,:]  = Xj; 
end

#optimization variables initial values
beta = A_train\y_train;
lambda = zeros(p);
for i in 1:p
	lambda[i] = std(A_train[:,p])
end

#standard error of regression estimate 
sigma = std(A_train*beta-y_train)

h = sigma*ones(p);


if(true)
	w = [log(sigma); log(h); log(lambda); beta]; 

	#first convert initial values to log-scales
	n_sample = 50; 

	for i in 1:1e2
		#compute gradient 
		#using only n_sample data points

		ind = randperm(n)[1:n_sample];
		data = A_train[ind,:];
		y = y_train[ind];

		#compute gradient
		grad = computeGradient(w,data, y, p); 

		println(w[2+2*p:end])

		w_prev = w; 
		#update parameters
		w = w - (1e-3)*(grad/norm(grad));
	end


	sigma = exp(w[1]);
	h = exp(w[2:2+p-1]);
	lambda = exp(w[2+p:2+2*p-1]);
	beta = w[2+2*p:end];
end



###############
###
###  Now that we've determined the "optimal hyperparameters", we do prediction 
###
###############

## We now "train" the model on the data

#set parameters 
T = size(A_train,1);
Sigma = eye(T)*sigma^2;

mu = Float64[];
for i in 1:p 
	for j in 1:T
		push!(mu, beta[i])
	end
end

n,m = size(A_train);
#package up the training features 
X = zeros(n*m, n); 
for j in 1:m
	Xj = A_train[:,j];
	Xj = diagm(Xj);
	loc = n*(j-1)+1:(n*j);
	X[loc,:]  = Xj; 
end

#generate training data covariance matrix
K = zeros(p*T, p*T);
for i in 1:p 
	Ki = generateCovarianceMatrix(h[i], lambda[i], A_train, A_train);
	loc = T*(i-1)+1:i*T;
	K[loc, loc] = Ki; 
end


#posterior mean and covariance of training data
post_mean = X'*mu + (Sigma + X'*K*X)\(X'*K*X)*(y_train-X'*mu);
post_cov = (Sigma + X'*K*X)\(X'*K*X)*Sigma;


#PLOT THE PREDICTION OF THE MODEL ON THE TRAINING DATA 
figure();
scatter(1:length(y_train), y_train, label = "true", color = "red")
y_trainvar = diag(post_cov);
errorbar(1:length(y_train), post_mean, y_trainvar.*1.96, label = "prediction");
ax = gca();
ax[:set_xlim]([0,T])
legend();
xlabel("Days");
ylabel("log(Price)");
savefig("figures/train_prediction.eps")


ax = gca();
ax[:set_xlim]([100,400])
legend();
savefig("figures/train_prediction_zoomed.eps")



##We now do prediction on test data

#package up the test features 
ntest, mtest = size(A_test);
Xtest = zeros(ntest*mtest, ntest);
for j in 1:mtest
	Xj = A_test[:,j];
	Xj = diagm(Xj);
	loc = ntest*(j-1)+1:(ntest*j);
	Xtest[loc,:] = Xj; 
end
Ttest = size(y_test,1);


#generate test data covariance matrix
Kstar = zeros(p*T, p*Ttest);
for i in 1:p 
	Ki = generateCovarianceMatrix(h[i], lambda[i], A_train, A_test);
	loc1 = T*(i-1)+1:i*T;
	loc2 = (Ttest*(i-1)+1):(i*Ttest)
	Kstar[loc1, loc2] = Ki; 
end
Kstarstar = zeros(p*Ttest, p*Ttest);
for i in 1:p
	Ki = generateCovarianceMatrix(h[i],lambda[i], A_test, A_test);
	loc = (Ttest*(i-1)+1):(i*Ttest);
	Kstarstar[loc, loc] = Ki;
end


mutest = Float64[];
for i in 1:p 
	for j in 1:Ttest
		push!(mutest, beta[i])
	end
end


#perform prediction
#calculate posterior mean and covariance
# of future observation 
y_pred = Xtest'*mutest + Xtest'*Kstar'*X*inv(X'*K*X + Sigma)*(y_train-X'*mu);
y_var = diag(Xtest'*(Kstarstar' - Kstar'*X*inv(X'*K*X + Sigma)*X'*Kstar)*Xtest .+sigma^2);


#plot
figure();
scatter(1:length(y_test),y_test, label = "true", color = "red")
errorbar(1:length(y_pred), y_pred, y_var.*1.96, label = "prediction");
legend();
ax = gca();
ax[:set_xlim]([0,Ttest])
xlabel("Days");
ylabel("log(Price)");
savefig("figures/test_prediction.eps")

ax = gca();
ax[:set_xlim]([100,200])
legend();
savefig("figures/test_prediction_zoomed.eps")





###############
###
###  In this section we perform sequential estimation, using the paramaters determined above 
###
###############


#run AR forward t_forward days in the future 
t_forward = 10; 
start = 250; 

#method 1
#run the full model from above forward

#first data point
curr_data = A_test[start,:]'; 
predictions1 = Float64[]; 
for(i in 1:t_forward)
	#generate next point covariance matrix
	Kstar_next = zeros(p*T, p);
	for i in 1:p 
		Ki = generateCovarianceMatrix(h[i], lambda[i], A_train, curr_data');
		loc1 = T*(i-1)+1:i*T;
		loc2 = ((i-1)+1):(i)
		Kstar_next[loc1, loc2] = Ki; 
	end
	#generate prediction 
	y_pred = curr_data'*beta + curr_data'*Kstar_next'*X*inv(X'*K*X + Sigma)*(y_train-X'*mu);
	push!(predictions1, y_pred[1])

	#shift everything over 1
	curr_data[2:p] = curr_data[1:(p-1)]
	#update current prediction
	curr_data[1] = y_pred[1];
end
figure();
scatter(1:t_forward, y_test[start:t_forward+start-1], label = "true");
plot(predictions1, label = "predictions");
legend(); 
xlabel("Days");
ylabel("log(price)");
savefig("figures/sequential_prediction1.eps")





#method2 
#assume posterior means are true and just sample from the noise 
#as you move forward

figure()
curr_data = A_test[start,:]'; 
srand(10); #the plot looks good with this random seed

predictions2 = Float64[]; 
for(i in 1:t_forward)
	y_pred = curr_data'*beta + rand(Normal(0,sigma));

	push!(predictions2, y_pred[1])
	#shift everything over 1
	curr_data[2:p] = curr_data[1:(p-1)]
	#update current prediction
	curr_data[1] = y_pred[1];
end
figure();
scatter(1:t_forward, y_test[start:t_forward+start-1], label = "true");
plot(predictions2, label = "predictions")
legend(); 
xlabel("Days");
ylabel("log(price)");
savefig("figures/sequential_prediction2.eps")














