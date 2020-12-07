function [L,k,b]=fitGLM_template(Spikes,X,fname)
    % INPUTS
    % Spikes:   row vector of length N with numbers of spikes at each time step
    % X:        raw (unfiltered) input. Matrix with dimensions DxN where D is
    %           the dimension of the input
    % fname:    a string indicating the nonlinearity to use 'quad', 'smooth', or 'exp'
    %
    % OUTPUTS
    % L:        row vector of loglikelihoods at every iteration of Newton-Raphson
    % k:        input filter coefficients (dimensionless)
    % b:        background firing rate Hz
    
    %% Some initialization
    % Choice of nonlinearity and its derivatives
    switch fname
        case 'exp'
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % the exponential and its derivatives and inverse
            f = @exp;
            df = @exp;
            d2f = @exp;
            invf = @log;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        case 'smooth'
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % the smooth rectified linear and its derivatives and inverse
            'your code goes here';
            % f=
            % df=
            % d2f=
            % invf= This one isn't technically needed, but it's nice to have for initialization of b (see below)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        case 'quad'
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % the rectified quadratic and its derivatives and inverse
            'your code goes here';
            % f=
            % df=
            % d2f=
            % invf= This one isn't technically needed, but it's nice to have for initialization of b (see below)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    % lines for local testing
    f = @exp;
    df = @exp;
    d2f = @exp;
    invf = @log;
    Spikes = Spikes_train;
    X = X_train;
    
    % Time step size
    dt=.001; % seconds
    
    % Tolerance for convergence
    tol=1e-7;
    
    % A good initialization for k is the mean of X at spike times
    k=mean(X(1:end,Spikes~=0),2);
    
    % A good initialization for b is the total average firing rate
    b=sum(Spikes)/(dt*length(Spikes));
    
    % Augment X with 1's so we maximize k and b together
    X=[X;ones(1,size(X,2))];
    
    % Invert f to augument k with b
    k=[k;invf(b)]; % Only if you know invf
    %k=[k;1]; % Or try this one

    % So we can watch k evolve...
    fig=figure;
    img=imagesc(reshape(k(1:end-1),7,7));
    
    % Initialize L at -inf
    L=-inf*ones(1,1000);
    %L=0*ones(1,1000);
    
    % Nonzero spike times (you'll want this!)
    nz=Spikes>0;
    total_t = 200000;
    
    % Loop until Newton-Raphson converges
    loop=1;
    k_old=k;
    while true
        loop=loop+1;
        
        % Trick to deal with bad Newton steps
        k=2*k-k_old; % initialize k such that when we take the mean in 2 lines it's the correct value
        while ~isfinite(L(loop))
            k=.5*(k+k_old); % move k back toward k_old
    
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Calculate log-likelihood (you can ignore the constant terms here if you want)
            ll = 0;
            for t = 1:total_t
                lambda_t = f(dot(k, X(:,t))); % don't add b because it's already concatenated within k
                yt = Spikes(t);
                %ll = ll + (yt * log(dt) + yt*log(lambda_t) - dt * lambda_t - log(factorial(yt)));
                ll = ll + (yt * log(lambda_t * dt) - dt * lambda_t - log(factorial(yt)));
            end
            L(loop) = ll;

            %L(loop) = Spiking_LL(X, k, Spikes, f);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
        
        % Plot k
        set(img,'CData',reshape(k(1:end-1),7,7));
        drawnow;

        % Calculate the size of the error
        delta=abs((L(loop)-L(loop-1))/L(loop));
        
        % Print iteration information
        fprintf('iter: %d, LL: %g, delta: %g\n',loop-1,L(loop),delta);
        
        % Convergence criteria
        if delta < tol
            break;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Calculate gradient
        disp('Calculating gradient')
        grad = zeros(50, 1); % gradient is vector of partials
        for j=1:size(grad)
            kj = 0;
            for t=1:total_t
                yt = Spikes(t);
                ui = dot(k, X(:, t));
                kj = kj + (yt * df(ui) / f(ui) - dt * df(ui)) * X(j, t);
            end

            % display update
            grad(j) = kj;
            if mod(j, 10) == 0
                disp(j);
            end
        end
        g = grad;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Calculate Hessian
        % yeesh loop city this hurts to look at
        % hessian is symmetric so all we need to do is fill in half
        disp('Calculating Hessian')
        hess = zeros(50);
        for j=1:size(hess)
            for m = j:size(hess)
                kij = 0;
                for t = 1:total_t
                    ui = dot(k, X(:, t));
                    kij = kij + ((((yt * d2f(ui) * f(ui)) - df(ui)^2) / f(ui)^2 - dt * d2f(ui)) * X(j, t) * X(m, t));
                end
            end
            hess(j, m) = kij;
            
            % display update to user
            if mod(j, 5) == 0
                disp(j);
            end
        end
        H = hess;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Newton-Raphson
        k_old=k;
        k = k_old - (H * g);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Deal the outputs
    %L=L(1:j);
    %b = k(50);
    %k = k(1:49);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Close the figure
    %close(fig);
end