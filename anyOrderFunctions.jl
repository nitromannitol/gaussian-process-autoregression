##############################################################################
# This code contains all the helper functions used in anyOrderFar.jl 
#
# This material was developed as a final project for Stats 370
# Bayesian Statistics
##############################################################################


#/*---------------------------------------------------------------------
# |  Function: compute[L or H]Gradient(h,lambda, data)
# |
# |  Purpose: Compute the gradient of the covariacne matrix with respect 
# |      		to log(lambda) and log(h)
# |      		The covariance function is C(x_1, x_2) = h^2 exp(-\|x1 - x2\|^2/(2lambda^2))
# |      
# |  Parameters:
# |			h - scalar corresponding to output scale 
# |			lambda - scalar corresponding to input scale
# |			data - input feature matrix
# |
# |  Returns:  The gradient of the covariance matrix with respect to h or lambda 
# |     
# *-------------------------------------------------------------------*/
function computeLGradient(h, lambda, data)
	t = size(data,1);
	#gradient with respect to lambda
	K_temp = zeros(t,t);
	for i in 1:t
		for j in 1:i
			x1 = data[i,:];
			x2 = data[j,:];
			temp = 2.0*h^2*exp(-0.5*norm(x1-x2)^2/lambda^2);
			K_temp[i,j] = temp;
			K_temp[j,i] = temp; 
		end
	end
	return K_temp;
end

function computeHGradient(h,lambda, data)
	#graident with respect to h
	t = size(data,1);
	K_temp = zeros(t,t);
	for i in 1:t
		for j in 1:i
			x1 = data[i,:];
			x2 = data[j,:];
			temp = h^2*norm(x1-x2)^2/lambda^2*exp(-0.5*norm(x1-x2)^2/lambda^2)
			K_temp[i,j] = temp;
			K_temp[j,i] = temp; 
		end
	end
	return K_temp;
end





#/*---------------------------------------------------------------------
# |  Function: computeGradient(w, data, y, p)
# |
# |  Purpose:  Computes the gradient of w given the data and y. See page 
# |      	   XXX in paper for details. 
# |
# |  Parameters:
# |      w - a vector with entries [log(sigma); log(h); log(lambda); beta]
# | 		  where h,lambda,beta are vectors of length p and sigma is a scalar
# |      data - matrix of input vectors        
# |      y - matrix of output vectors                
# |      p  - autoregressive window                 
# |                       
# |  Returns:  The gradient vector evaluated at w of length, length(w)
# |      
# *-------------------------------------------------------------------*/
function computeGradient(w, data, y, p)
	T = size(data,1);
	n,m = size(data);
	X = zeros(n*m, n); 

	#define necessary matrices and vectors
	for j in 1:m
		Xj = data[:,j];
		Xj = diagm(Xj);
		loc = n*(j-1)+1:(n*j);
		X[loc,:]  = Xj; 
	end

	#take exponent since all but beta are in log form
	sigma = exp(w[1]);
	h = exp(w[2:2+p-1]);
	lambda = exp(w[2+p:2+2*p-1]);
	beta = w[2+2*p:end];

	Sigma = eye(T)*sigma^2;

	mu = Float64[];
	for i in 1:p 
		for j in 1:T
			push!(mu, beta[i])
		end
	end

	#build block-diagonal covariance matrix 
	K = zeros(p*T, p*T);
	for i in 1:p 
		Ki = generateCovarianceMatrix(h[i], lambda[i], data, data);
		loc = T*(i-1)+1:i*T;
		K[loc, loc] = Ki; 
	end

	#cache intermediate values
	S = X'*K*X + Sigma; 
	_w = S\(y-X'*mu);

	#calculate gradients for beta
	dmu_dbeta = zeros(p);
	for i in 1:p 
		z = zeros(p*T);
		z[((i-1)*T +1):(i*T)] = ones(T);
		dmu_dbeta[i] = ((y-X'*mu)'*inv(S)*X'*z)[1];
	end



	#gradient with respect to log(sigma) 
	S_sigmaprime = 2*sigma^2*eye(T);
	dl_dsigma = 0.5*trace((_w*_w' - inv(S))*(S_sigmaprime));


	#gradients with respect to log(lambda_i)
	dl_dlambda = zeros(p);
	for i in 1:p 
		S_lambdaprimei = zeros(p*T, p*T);
		loc = ((i-1)*T +1):(i*T);
		S_lambdaprimei[loc,loc] = computeLGradient(h[i], lambda[i], data);
		dl_dlambda[i] = 0.5*trace((_w*_w' - inv(S))*(X'*S_lambdaprimei*X));
	end


	#gradients of S with respect to log(h_i)
	dl_dh = zeros(p);
	for i in 1:p
		S_hprimei = zeros(p*T, p*T)
		loc = ((i-1)*T +1):(i*T);
		S_hprimei[loc,loc] = computeHGradient(h[i], lambda[i], data);
		dl_dh[i] = 0.5*trace((_w*_w' - inv(S))*(X'*S_hprimei*X));
	end

	#comment out if optimizing over the h 
	dl_dh = zeros(p);

	#compute marginal log-likelihood
	#meanVal = (y-X'*mu);
	#loglikelihood = logdet(X'*K*X + Sigma) - meanVal'*inv(X'*K*X + Sigma)*meanVal;
	#println(string("LL: ", loglikelihood[1])); 

	#package up the gradients and return them
	return [dl_dsigma; dl_dh; dl_dlambda; dmu_dbeta];
end




#/*---------------------------------------------------------------------
# |  Function: generateCovarianceMatrix(h,lambda, X1, X2)
# |
# |  Purpose: Computes the covariance matrix C(X1, X2)
# |      	  where C(x_1, x_2) = h^2 exp(-\|x1 - x2\|^2/(2lambda^2))
# |
# |  Parameters:
# |       h - scalar corresponding to output scale 
# |       lambda- scalar corresponding to input scale       
# |       X1 - matrix vector of data              
# |       X2 - matrix vector of data                
# |                       
# |  Returns:  Returns K, T1 x T2 matrix of covariances between X1 and X2
# |    
# *-------------------------------------------------------------------*/
function generateCovarianceMatrix(h,lambda, X1, X2)
	T1 = size(X1,1);
	T2 = size(X2,1);
	K = zeros(T1,T2);

	for i in 1:T1
		for j in 1:T2
			x1 = X1[i,:];
			x2 = X2[j,:];
			K[i,j] = h^2*exp(-norm(x1 - x2)^2/(2*(lambda)^2));
		end
	end
	return K;
end



#/*---------------------------------------------------------------------
# |  Function: toepltiz(vect,n) 
# |
# |  Purpose: Builds the autoregression toeplitz matrix from 
# |      	  vector vect with window n
# |
# |  Parameters:
# |      	vect - a vector (of length larger than n)
# |         n - the autoregressive window      
# |                                            
# |  Returns:  The toepltiz matrix T of size m+n-1 by n 
# | 
# *-------------------------------------------------------------------*/
function toeplitz(vect,n)
	m = size(vect,1);
	T = zeros(m+n-1,n);
	for i in 1:n
		T[i:(i+m-1),i] = vect;
	end
	return T
end



